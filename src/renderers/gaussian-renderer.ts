import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_scale: (scale: number) => void,
  set_culling: (enabled: boolean) => void,
  set_workgroup_size: (size: number) => void,
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: GPUAllowSharedBufferSource
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);
  const dispatch_data = new Uint32Array([0, 1, 1]);
  // Center, quad size, conic, and color occupy 64 bytes with storage alignment.
  const splat_buffer = createBuffer(device, '2d splats', Math.max(1, pc.num_points) * 64,
    GPUBufferUsage.STORAGE);
  const settings_data = new Float32Array([1, pc.sh_deg, 1, 0]);
  const settings_buffer = createBuffer(device, 'render settings', 16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, settings_data);
  const draw_buffer = createBuffer(device, 'quad draw indirect', 16,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, new Uint32Array([4, 0, 0, 0]));

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  let workgroup_size = C.histogram_wg_size;
  const preprocess_shader = device.createShaderModule({ code: preprocessWGSL });
  const binding_types: GPUBufferBindingType[][] = [
    ['uniform', 'uniform'],
    ['read-only-storage', 'storage', 'read-only-storage'],
    ['storage', 'storage', 'storage', 'storage'],
  ];
  const preprocess_layout = device.createPipelineLayout({
    bindGroupLayouts: binding_types.map(types => device.createBindGroupLayout({
      entries: types.map((type, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type },
      })),
    })),
  });
  let preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: preprocess_layout,
    compute: {
      module: preprocess_shader,
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortBlockSize: C.histogram_wg_size * c_histogram_block_rows,
      },
    },
  });

  const preprocess_pipelines = new Map([[workgroup_size, preprocess_pipeline]]);

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess camera and settings',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: settings_buffer } },
    ],
  });
  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({ code: renderWGSL });
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian quads',
    layout: 'auto',
    vertex: { module: render_shader, entryPoint: 'vs_main' },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-strip' },
  });
  const render_bind_group = device.createBindGroup({
    label: 'quad splats and indices',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, dispatch_data);
      const compute_pass = encoder.beginComputePass({ label: 'preprocess' });
      compute_pass.setPipeline(preprocess_pipeline);
      compute_pass.setBindGroup(0, camera_bind_group);
      compute_pass.setBindGroup(1, gaussian_bind_group);
      compute_pass.setBindGroup(2, sort_bind_group);
      compute_pass.dispatchWorkgroups(Math.ceil(pc.num_points / workgroup_size));
      compute_pass.end();

      // Copy the visible count into instanceCount, after preprocessing finishes.
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, draw_buffer, 4, 4);
      sorter.sort(encoder);
      const pass = encoder.beginRenderPass({
        label: 'gaussian quad render',
        colorAttachments: [{
          view: texture_view,
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(render_pipeline);
      pass.setBindGroup(0, render_bind_group);
      pass.drawIndirect(draw_buffer, 0);
      pass.end();
    },
    set_scale: (scale: number) => {
      settings_data[0] = scale;
      device.queue.writeBuffer(settings_buffer, 0, settings_data);
    },
    set_culling: (enabled: boolean) => {
      settings_data[2] = enabled ? 1 : 0;
      device.queue.writeBuffer(settings_buffer, 0, settings_data);
    },
    set_workgroup_size: (size: number) => {
      if (![64, 128, 256].includes(size)) {
        throw new Error('Preprocess workgroup size must be 64, 128, or 256.');
      }
      if (!preprocess_pipelines.has(size)) {
        preprocess_pipelines.set(size, device.createComputePipeline({
          label: `preprocess ${size}`,
          layout: preprocess_layout,
          compute: {
            module: preprocess_shader,
            entryPoint: 'preprocess',
            constants: {
              workgroupSize: size,
              // Sorting still uses its original 256-thread groups.
              sortBlockSize: C.histogram_wg_size * c_histogram_block_rows,
            },
          },
        }));
      }
      workgroup_size = size;
      preprocess_pipeline = preprocess_pipelines.get(size)!;
    },
    camera_buffer,
  };
}
