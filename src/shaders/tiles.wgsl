struct Splat {
    center: vec4<f32>, half_size: vec2<f32>, radius: f32,
    conic: vec4<f32>, color: vec4<f32>,
}
struct Grid { width: u32, height: u32, columns: u32, rows: u32 }
@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> depths: array<u32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;
@group(0) @binding(4) var<uniform> grid: Grid;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> entries: array<vec2<u32>>;
@group(0) @binding(7) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var<storage, read_write> sort_dispatch: array<u32>;
@group(0) @binding(9) var<storage, read_write> fallback_draw: array<u32>;
@group(0) @binding(10) var fallback_image: texture_2d<f32>;
const capacity = 1024u;
const tile_size = 16u;
fn pixel_center(s: Splat) -> vec2<f32> {
    return (s.center.xy * vec2<f32>(0.5, -0.5) + 0.5) * vec2<f32>(f32(grid.width), f32(grid.height));
}
@compute @workgroup_size(256)
fn bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= info[0]) { return; }
    let index = indices[gid.x];
    let s = splats[index];
    if (s.center.z < 0. || s.center.z > 1.) { return; }
    let center = pixel_center(s);
    let lo = max(vec2<i32>(0), vec2<i32>(floor((center - s.radius) / 16.)));
    let hi = min(vec2<i32>(i32(grid.columns)-1, i32(grid.rows)-1), vec2<i32>(floor((center + s.radius) / 16.)));
    for (var y=lo.y; y<=hi.y; y++) {
        for (var x=lo.x; x<=hi.x; x++) {
            let tile = u32(y) * grid.columns + u32(x);
            let slot = atomicAdd(&counts[tile], 1u);
            if (slot == capacity) { atomicStore(&counts[grid.columns * grid.rows], 1u); }
            if (slot < capacity) { entries[tile * capacity + slot] = vec2<u32>(depths[gid.x], index); }
        }
    }
}
// Only overflow frames need the global radix sort.
@compute @workgroup_size(1)
fn prepare_fallback() {
    let overflow = atomicLoad(&counts[grid.columns * grid.rows]) != 0u;
    sort_dispatch[0] = select(0u, (info[0] + 3839u) / 3840u, overflow);
    fallback_draw[0] = 4u;
    fallback_draw[1] = select(0u, info[0], overflow);
    fallback_draw[2] = 0u;
    fallback_draw[3] = 0u;
}
var<workgroup> keys: array<vec2<u32>, 1024>;
fn greater(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x > b.x || (a.x == b.x && a.y > b.y);
}
@compute @workgroup_size(256)
fn sort_tiles(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let tile = group.y * grid.columns + group.x;
    let count = min(atomicLoad(&counts[tile]), capacity);
    for (var i=tid; i<capacity; i+=256u) {
        keys[i] = vec2<u32>(0xffffffffu);
        if (i<count) { keys[i] = entries[tile*capacity+i]; }
    }
    workgroupBarrier();
    for (var k=2u; k<=capacity; k*=2u) {
        for (var j=k/2u; j>0u; j/=2u) {
            for (var i=tid; i<capacity; i+=256u) {
                let other=i^j;
                if (other>i) {
                    let a=keys[i]; let b=keys[other];
                    if (greater(a,b) == ((i&k)==0u)) { keys[i]=b; keys[other]=a; }
                }
            }
            workgroupBarrier();
        }
    }
    for (var i=tid; i<count; i+=256u) { entries[tile*capacity+i]=keys[i]; }
}
@compute @workgroup_size(8,8)
fn composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x>=grid.width || gid.y>=grid.height) { return; }
    let tile=(gid.y/tile_size)*grid.columns+gid.x/tile_size;
    let count=atomicLoad(&counts[tile]);
    if (count>capacity) {
        textureStore(output,vec2<i32>(gid.xy),textureLoad(fallback_image,vec2<i32>(gid.xy),0));
        return;
    }
    let length=count;
    let pixel=vec2<f32>(gid.xy)+0.5;
    var color=vec3<f32>(0.); var transmittance=1.;
    // Near to far allows fully covered pixels to stop early.
    for (var n=length; n>0u; n--) {
        let index=entries[tile*capacity+n-1u].y;
        let s=splats[index];
        if (s.center.z<0. || s.center.z>1.) { continue; }
        let d=(pixel-pixel_center(s))*vec2<f32>(1.,-1.);
        if (any(abs(d)>vec2<f32>(s.radius))) { continue; }
        let power=-0.5*(s.conic.x*d.x*d.x+2.*s.conic.y*d.x*d.y+s.conic.z*d.y*d.y);
        let alpha=min(0.99,s.color.a*exp(power));
        if (power>0. || alpha<1./255.) { continue; }
        color+=transmittance*alpha*s.color.rgb;
        transmittance*=1.-alpha;
        if (transmittance<0.0001) { break; }
    }
    textureStore(output,vec2<i32>(gid.xy),vec4<f32>(color,1.));
}
