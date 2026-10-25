import tilesWGSL from '../shaders/tiles.wgsl';
import gaussianWGSL from '../shaders/gaussian.wgsl';
import presentWGSL from '../shaders/tile-present.wgsl';
import { SortStuff } from '../sort/sort';

export default function get_tile_renderer(device: GPUDevice, format: GPUTextureFormat,
  splats: GPUBuffer, sorter: SortStuff, point_count: number) {
  const module = device.createShaderModule({ code: tilesWGSL });
  const layout = device.createBindGroupLayout({ entries: [
    ...['read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage',
      'uniform', 'storage', 'storage'].map((type, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: type as GPUBufferBindingType },
    })),
    { binding: 7, visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: {} },
  ] });
  const pipeline_layout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipelines = ['bin', 'sort_tiles', 'composite', 'prepare_fallback'].map(entryPoint =>
    device.createComputePipeline({ layout: pipeline_layout, compute: { module, entryPoint } }));
  const present_module = device.createShaderModule({ code: presentWGSL });
  const present = device.createRenderPipeline({ layout: 'auto',
    vertex: { module: present_module, entryPoint: 'vs_main' },
    fragment: { module: present_module, entryPoint: 'fs_main', targets: [{ format }] },
  });
  const fallback_module = device.createShaderModule({ code: gaussianWGSL.replace(
    '    let d = in.offset;',
    `    let pixel = vec2<u32>(in.position.xy);
    let tile = (pixel.y / 16u) * tile_grid.z + pixel.x / 16u;
    if (tile_counts[tile] <= 1024u) { discard; }
    let d = in.offset;`) + `
@group(0) @binding(2) var<storage, read> tile_counts: array<u32>;
@group(0) @binding(3) var<uniform> tile_grid: vec4<u32>;
` });
  const fallback_pipeline = device.createRenderPipeline({ layout: 'auto',
    vertex: { module: fallback_module, entryPoint: 'vs_main' },
    fragment: { module: fallback_module, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm',
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      },
    }] }, primitive: { topology: 'triangle-strip' },
  });
  const fallback_draw = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
  let fallback_image: GPUTexture, fallback_bindings: GPUBindGroup;
  const grid = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let width = 0, height = 0, columns = 0, rows = 0;
  let warned = false;
  let counts: GPUBuffer, entries: GPUBuffer, output: GPUTexture;
  let bindings: GPUBindGroup, present_bindings: GPUBindGroup;
  function resize(w: number, h: number) {
    if (w === width && h === height) return true;
    const cols = Math.ceil(w / 16), tile_rows = Math.ceil(h / 16);
    const bytes = cols * tile_rows * 1024 * 8;
    if (bytes > device.limits.maxStorageBufferBindingSize || bytes > device.limits.maxBufferSize) {
      if (!warned) console.warn('Tile storage exceeds the device limit; using global sorting for this viewport.');
      warned = true;
      return false;
    }
    counts?.destroy(); entries?.destroy(); output?.destroy(); fallback_image?.destroy();
    warned = false;
    width = w; height = h; columns = cols; rows = tile_rows;
    counts = device.createBuffer({ size: (columns * rows + 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    entries = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
    output = device.createTexture({ size: [width, height], format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    fallback_image = device.createTexture({ size: [width, height], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    fallback_bindings = device.createBindGroup({ layout: fallback_pipeline.getBindGroupLayout(0), entries:
      [splats, sorter.ping_pong[0].sort_indices_buffer, counts, grid].map((buffer, binding) => ({
        binding, resource: { buffer },
      })),
    });
    device.queue.writeBuffer(grid, 0, new Uint32Array([width, height, columns, rows]));
    bindings = device.createBindGroup({ layout, entries: [
      ...[splats, sorter.ping_pong[0].sort_indices_buffer, sorter.ping_pong[0].sort_depths_buffer,
        sorter.sort_info_buffer, grid, counts, entries].map((buffer, binding) => ({ binding, resource: { buffer } })),
      { binding: 7, resource: output.createView() },
      { binding: 8, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      { binding: 9, resource: { buffer: fallback_draw } },
      { binding: 10, resource: fallback_image.createView() },
    ] });
    present_bindings = device.createBindGroup({ layout: present.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: output.createView() }] });
    return true;
  }
  function dispatch(encoder: GPUCommandEncoder, stage: number, x: number, y = 1) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines[stage]); pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(x, y); pass.end();
  }
  return {
    resize,
    bin: (encoder: GPUCommandEncoder) => {
      encoder.clearBuffer(counts);
      dispatch(encoder, 0, Math.ceil(point_count / 256));
      dispatch(encoder, 1, columns, rows);
      dispatch(encoder, 3, 1);
    },
    frame: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
      const fallback_pass = encoder.beginRenderPass({ colorAttachments: [{ view: fallback_image.createView(),
        loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
      fallback_pass.setPipeline(fallback_pipeline); fallback_pass.setBindGroup(0, fallback_bindings);
      fallback_pass.drawIndirect(fallback_draw, 0); fallback_pass.end();
      dispatch(encoder, 2, Math.ceil(width / 8), Math.ceil(height / 8));
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear',
        storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(present); pass.setBindGroup(0, present_bindings); pass.draw(3); pass.end();
    },
  };
}
