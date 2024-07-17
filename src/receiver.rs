use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsage, CopyBufferToImageInfo, RecordingCommandBuffer,
    },
    device::{Device, Queue},
    image::{Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::GpuFuture,
    DeviceSize,
};
use wlx_capture::{frame::WlxFrame, pipewire::PipewireCapture, WlxCapture};

use crate::dmabuf::{dmabuf_texture, fourcc_to_vk};

pub fn try_receive_frame(
    rx: &mut PipewireCapture,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
) -> Option<Arc<Image>> {
    match rx.receive()? {
        WlxFrame::Dmabuf(frame) => dmabuf_texture(device, memory_allocator, frame).ok(),
        WlxFrame::MemFd(frame) => {
            let mut uploads = RecordingCommandBuffer::new(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::OneTimeSubmit,
                    ..Default::default()
                },
            )
            .unwrap();

            let fd = frame.plane.fd?;
            let format = fourcc_to_vk(frame.format.fourcc).ok()?;

            let len = frame.plane.stride as usize * frame.format.height as usize;
            let offset = frame.plane.offset as i64;

            let map = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    fd,
                    offset,
                )
            } as *const u8;

            let data = unsafe { std::slice::from_raw_parts(map, len) };

            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent: [frame.format.width, frame.format.height, 1],
                    usage: ImageUsage::TRANSFER_DST
                        | ImageUsage::TRANSFER_SRC
                        | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            let buffer: Subbuffer<[u8]> = Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                data.len() as DeviceSize,
            )
            .unwrap();

            buffer.write().unwrap().copy_from_slice(data);

            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer, image.clone()))
                .unwrap();

            uploads
                .end()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .flush()
                .unwrap();

            unsafe { libc::munmap(map as *mut _, len) };

            Some(image)
        }
        _ => None,
    }
}
