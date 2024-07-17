use std::{
    os::fd::{FromRawFd, IntoRawFd},
    sync::Arc,
};

use anyhow::{anyhow, bail};
use vulkano::{
    device::Device,
    format::Format,
    image::{sys::RawImage, Image, ImageCreateInfo, ImageTiling, ImageUsage, SubresourceLayout},
    memory::{
        allocator::{MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator},
        DedicatedAllocation, DeviceMemory, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        MemoryAllocateInfo, MemoryImportInfo, MemoryPropertyFlags, ResourceMemory,
    },
};
use wlx_capture::frame::{
    DmabufFrame, FourCC, DRM_FORMAT_ABGR2101010, DRM_FORMAT_ABGR8888, DRM_FORMAT_ARGB8888,
    DRM_FORMAT_XBGR2101010, DRM_FORMAT_XBGR8888, DRM_FORMAT_XRGB8888,
};

pub const DRM_FORMAT_MOD_INVALID: u64 = 0xff_ffff_ffff_ffff;

pub fn dmabuf_texture(
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    frame: DmabufFrame,
) -> anyhow::Result<Arc<Image>> {
    let extent = [frame.format.width, frame.format.height, 1];

    let format = fourcc_to_vk(frame.format.fourcc)?;

    let mut tiling: ImageTiling = ImageTiling::Optimal;
    let mut modifiers: Vec<u64> = vec![];
    let mut layouts: Vec<SubresourceLayout> = vec![];

    if frame.format.modifier != DRM_FORMAT_MOD_INVALID {
        (0..frame.num_planes).for_each(|i| {
            let plane = &frame.planes[i];
            layouts.push(SubresourceLayout {
                offset: plane.offset as _,
                size: 0,
                row_pitch: plane.stride as _,
                array_pitch: None,
                depth_pitch: None,
            });
            modifiers.push(frame.format.modifier);
        });
        tiling = ImageTiling::DrmFormatModifier;
    };

    let image = unsafe {
        RawImage::new_unchecked(
            device.clone(),
            ImageCreateInfo {
                format,
                extent,
                usage: ImageUsage::SAMPLED,
                external_memory_handle_types: ExternalMemoryHandleTypes::DMA_BUF,
                tiling,
                drm_format_modifiers: modifiers,
                drm_format_modifier_plane_layouts: layouts,
                ..Default::default()
            },
        )?
    };

    let requirements = image.memory_requirements()[0];
    let memory_type_index = memory_allocator
        .find_memory_type_index(
            requirements.memory_type_bits,
            MemoryTypeFilter {
                required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
        )
        .ok_or_else(|| anyhow!("failed to get memory type index"))?;

    debug_assert!(device.enabled_extensions().khr_external_memory_fd);
    debug_assert!(device.enabled_extensions().khr_external_memory);
    debug_assert!(device.enabled_extensions().ext_external_memory_dma_buf);

    // only do the 1st
    unsafe {
        let Some(fd) = frame.planes[0].fd else {
            bail!("DMA-buf plane has no FD");
        };

        let file = std::fs::File::from_raw_fd(fd);
        let new_file = file.try_clone()?;
        file.into_raw_fd();

        let memory = DeviceMemory::allocate_unchecked(
            device.clone(),
            MemoryAllocateInfo {
                allocation_size: requirements.layout.size(),
                memory_type_index,
                dedicated_allocation: Some(DedicatedAllocation::Image(&image)),
                ..Default::default()
            },
            Some(MemoryImportInfo::Fd {
                file: new_file,
                handle_type: ExternalMemoryHandleType::DmaBuf,
            }),
        )?;

        let mem_alloc = ResourceMemory::new_dedicated(memory);
        match image.bind_memory_unchecked([mem_alloc]) {
            Ok(image) => Ok(Arc::new(image)),
            Err(e) => {
                bail!("Failed to bind memory to image: {}", e.0);
            }
        }
    }
}

#[allow(non_snake_case)]
pub fn fourcc_to_vk(fourcc: FourCC) -> anyhow::Result<Format> {
    match fourcc.value {
        DRM_FORMAT_ABGR8888 => Ok(Format::R8G8B8A8_UNORM),
        DRM_FORMAT_XBGR8888 => Ok(Format::R8G8B8A8_UNORM),
        DRM_FORMAT_ARGB8888 => Ok(Format::B8G8R8A8_UNORM),
        DRM_FORMAT_XRGB8888 => Ok(Format::B8G8R8A8_UNORM),
        DRM_FORMAT_ABGR2101010 => Ok(Format::A2B10G10R10_UNORM_PACK32),
        DRM_FORMAT_XBGR2101010 => Ok(Format::A2B10G10R10_UNORM_PACK32),
        _ => bail!("Unsupported format {}", fourcc),
    }
}
