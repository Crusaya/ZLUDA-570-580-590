extern crate lazy_static;
#[cfg(test)]
extern crate paste;
extern crate ptx;

#[allow(warnings)]
pub mod cuda;
pub(crate) mod r#impl;

use crate::r#impl::LiveCheck;
use cuda_types::CUresult;
use hip_common::zluda_ext::{CudaObjectKind, CudaResult};
use r#impl::{context, stream};

const DRIVER_VERSION: i32 = 12040;
const CUDA_13_DRIVER_VERSION: i32 = 13000;

fn driver_version_for_cuda13(experimental_cuda_13: bool) -> i32 {
    if experimental_cuda_13 {
        CUDA_13_DRIVER_VERSION
    } else {
        DRIVER_VERSION
    }
}

pub(crate) fn driver_version() -> i32 {
    let experimental_cuda_13 =
        matches!(std::env::var("ZLUDA_EXPERIMENTAL_CUDA_13").as_deref(), Ok("1"));
    driver_version_for_cuda13(experimental_cuda_13)
}

#[no_mangle]
pub unsafe extern "C" fn zluda_get_hip_object(
    cuda_object: *mut std::os::raw::c_void,
    kind: CudaObjectKind,
) -> CudaResult<*const std::os::raw::c_void> {
    unsafe fn zluda_get_hip_object_impl(
        cuda_object: *const std::os::raw::c_void,
        kind: CudaObjectKind,
    ) -> Result<*const std::os::raw::c_void, CUresult> {
        match kind {
            CudaObjectKind::Context => {
                let cuda_object = cuda_object as *mut context::Context;
                let ctx = LiveCheck::as_result(cuda_object)?;
                Ok(ctx.device as usize as _)
            }
            CudaObjectKind::Stream => {
                let cuda_object = cuda_object as *mut stream::Stream;
                let stream = stream::as_hip_stream(cuda_object)?;
                Ok(stream as _)
            }
        }
    }
    zluda_get_hip_object_impl(cuda_object, kind).into()
}

#[no_mangle]
pub unsafe extern "C" fn zluda_get_version() -> *const std::os::raw::c_char {
    return env!("VERGEN_GIT_SHA").as_ptr() as _;
}

#[no_mangle]
pub unsafe extern "C" fn zluda_get_nightly_flag() -> std::os::raw::c_int {
    #[cfg(feature = "nightly")]
    return 1;
    #[cfg(not(feature = "nightly"))]
    return 0;
}

#[cfg(test)]
mod tests {
    #[test]
    fn experimental_cuda13_reports_cuda13_driver_version() {
        assert_eq!(super::driver_version_for_cuda13(false), 12040);
        assert_eq!(super::driver_version_for_cuda13(true), 13000);
    }
}
