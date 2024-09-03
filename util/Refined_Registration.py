import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Define the paths
hema_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\HE_gray'
dapi_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\dapi'
cd20_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\cd20'
cd4_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\cd4'
bcl2_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\bcl2'
irf4_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\irf4'
cd15_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\cd15'
pax5_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\pax5'
pd1_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\pd1'
he_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_first_reg\HE'
he_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\Registered_HE'
dapi_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_dapi'
cd20_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_cd20'
cd4_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_cd4'
bcl2_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_bcl2'
irf4_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_irf4'
cd15_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_cd15'
pax5_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_pax5'
pd1_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_2nd_reg\registered_pd1'

# Create the output directories if they don't exist
output_paths = [he_output_path, dapi_output_path, cd20_output_path, cd4_output_path, bcl2_output_path,
                irf4_output_path, cd15_output_path, pax5_output_path, pd1_output_path]
for path in output_paths:
    os.makedirs(path, exist_ok=True)

# Define the iteration callback function
def command_iteration(method):
    print(f"Iteration: {method.GetOptimizerIteration()}")
    print(f"Metric Value: {method.GetMetricValue()}")

# Function to extract the common prefix from the filename
def extract_prefix(filename):
    return filename.split('.')[0].replace('final_HE_', '')

# Function to perform affine registration
def affine_registration(fixed_image, moving_image):
    print("Performing affine registration")
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=150)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1,
        minStep=1e-4,
        numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    final_transform = registration_method.Execute(fixed_image, moving_image)
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(f"Optimizer converged: {registration_method.GetOptimizerConvergenceValue()}")
    return final_transform

# Function to perform B-spline registration
def bspline_registration(fixed_image, moving_image, bspline_grid_size):
    print("Performing B-spline registration")
    initial_transform = sitk.BSplineTransformInitializer(fixed_image, bspline_grid_size)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.01,
        minStep=1e-4,
        numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    final_transform = registration_method.Execute(fixed_image, moving_image)
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(f"Optimizer converged: {registration_method.GetOptimizerConvergenceValue()}")
    return final_transform

# Function to perform demons registration
def demons_registration(fixed_image, moving_image):
    print("Performing demons registration")
    demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(50)
    demons_filter.SetStandardDeviations(1.0)

    displacement_field = demons_filter.Execute(fixed_image, moving_image)
    final_transform = sitk.DisplacementFieldTransform(displacement_field)

    print(f"Final demons metric value: {demons_filter.GetMetric()}")
    return final_transform

# Function to perform the appropriate registration based on the method selected
def perform_registration(fixed_image, moving_image, transform_type="affine", bspline_grid_size=[4, 4]):
    fixed_image_float32 = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image_float32 = sitk.Cast(moving_image, sitk.sitkFloat32)

    if transform_type == "affine":
        return affine_registration(fixed_image_float32, moving_image_float32)
    elif transform_type == "bspline":
        return bspline_registration(fixed_image_float32, moving_image_float32, bspline_grid_size)
    elif transform_type == "demons":
        return demons_registration(fixed_image_float32, moving_image_float32)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

# Function to apply the transform and save the output
def apply_transform_and_save(image_path, output_path, transform, fixed_image):
    image = sitk.ReadImage(image_path, sitk.sitkUInt8)
    transformed_image = sitk.Resample(image, fixed_image, transform, sitk.sitkLinear, 0.0, image.GetPixelIDValue())
    sitk.WriteImage(transformed_image, output_path)

# Perform registration and apply the transform to all images
for he_file in os.listdir(hema_path):
    if he_file.endswith('.tif'):
        prefix = extract_prefix(he_file)
        dapi_file = f"final_mIHC_{prefix}_channel_1.tif"
        cd20_file = f"final_mIHC_{prefix}_channel_2.tif"
        cd4_file = f"final_mIHC_{prefix}_channel_4.tif"
        bcl2_file = f"final_mIHC_{prefix}_channel_5.tif"
        irf4_file = f"final_mIHC_{prefix}_channel_6.tif"
        cd15_file = f"final_mIHC_{prefix}_channel_7.tif"
        pax5_file = f"final_mIHC_{prefix}_channel_9.tif"
        pd1_file = f"final_mIHC_{prefix}_channel_10.tif"
        he_file = f"final_HE_{prefix}.tif"

        hema_image_path = os.path.join(hema_path, he_file)
        dapi_image_path = os.path.join(dapi_path, dapi_file)
        cd20_image_path = os.path.join(cd20_path, cd20_file)
        cd4_image_path = os.path.join(cd4_path, cd4_file)
        bcl2_image_path = os.path.join(bcl2_path, bcl2_file)
        irf4_image_path = os.path.join(irf4_path, irf4_file)
        cd15_image_path = os.path.join(cd15_path, cd15_file)
        pax5_image_path = os.path.join(pax5_path, pax5_file)
        pd1_image_path = os.path.join(pd1_path, pd1_file)
        he_image_path = os.path.join(he_path, he_file)

        # Read the images as uint8
        print(f'Reading images for {he_file}')
        fixed_image = sitk.ReadImage(hema_image_path, sitk.sitkUInt8)
        fixed_image = sitk.InvertIntensity(fixed_image, maximum=255)
        moving_image = sitk.ReadImage(dapi_image_path, sitk.sitkUInt8)
        he_image = sitk.ReadImage(he_image_path, sitk.sitkVectorUInt8)

        # Choose the type of registration (affine, bspline, demons)
        transform_type = "demons"  # Change this to "bspline" or "demons" as needed
        bspline_grid_size = [4, 4]  # Set the B-spline grid size if using B-spline

        # Perform the registration
        final_transform = perform_registration(fixed_image, moving_image, transform_type, bspline_grid_size)

        # Apply the transformation to the DAPI, CD20, CD4, BCL2, IRF4, CD15, PAX5, and PD1 images
        apply_transform_and_save(dapi_image_path, os.path.join(dapi_output_path, f"{prefix}.tif"), final_transform, fixed_image)
        apply_transform_and_save(cd20_image_path, os.path.join(cd20_output_path, f"{prefix}.tif"), final_transform, fixed_image)
        apply_transform_and_save(cd4_image_path, os.path.join(cd4_output_path, f"{prefix}.tif"), final_transform, fixed_image)
        apply_transform_and_save(bcl2_image_path, os.path.join(bcl2_output_path, f"{prefix}.tif"), final_transform, fixed_image)
        apply_transform_and_save(irf4_image_path, os.path.join(irf4_output_path, f"{prefix}.tif"), final_transform, fixed_image)
        apply_transform_and_save(cd15_image_path, os.path.join(cd15_output_path, f"{prefix}.tif"), final_transform, fixed_image)
        apply_transform_and_save(pax5_image_path, os.path.join(pax5_output_path, f"{prefix}.tif"), final_transform, fixed_image)
        apply_transform_and_save(pd1_image_path, os.path.join(pd1_output_path, f"{prefix}.tif"), final_transform, fixed_image)

        # Save the HE image (unaltered)
        sitk.WriteImage(he_image, os.path.join(he_output_path, f"{prefix}.tif"))

print("Processing complete.")
