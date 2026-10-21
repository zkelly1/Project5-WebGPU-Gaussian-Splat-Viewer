struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

struct Splat {
    center: vec4<f32>,
    half_size: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let splat = splats[sort_indices[instance_idx]];
    // The four corners form two triangles in a strip.
    let corner = vec2<f32>(f32(vertex_idx & 1u), f32(vertex_idx >> 1u)) * 2. - 1.;
    var out: VertexOutput;
    out.position = splat.center + vec4<f32>(corner * splat.half_size, 0., 0.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}
