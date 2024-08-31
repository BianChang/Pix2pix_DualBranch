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
os.makedirs(he_output_path, exist_ok=True)
os.makedirs(dapi_output_path, exist_ok=True)
os.makedirs(cd20_output_path, exist_ok=True)
os.makedirs(cd4_output_path, exist_ok=True)
os.makedirs(bcl2_output_path, exist_ok=True)
os.makedirs(irf4_output_path, exist_ok=True)
os.makedirs(cd15_output_path, exist_ok=True)
os.makedirs(pax5_output_path, exist_ok=True)
os.makedirs(pd1_output_path, exist_ok=True)

# Define the iteration callback function
def command_iteration(method):
    print(f"Iteration: {method.GetOptimizerIteration()}")
    print(f"Metric Value: {method.GetMetricValue()}")

# Function to extract the common prefix from the filename
def extract_prefix(filename):
    return filename.split('.')[0].replace('final_HE_', '')

# Function to crop the image by specified coordinates
def crop_image_by_coordinates(image, top_left, bottom_right):
    size = [bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]]
    cropped_image = sitk.RegionOfInterest(image, size=size, index=top_left)
    return cropped_image

# Function to handle the rectangle selector
def line_select_callback(eclick, erelease):
    global top_left, bottom_right
    top_left = (int(eclick.xdata), int(eclick.ydata))
    bottom_right = (int(erelease.xdata), int(erelease.ydata))

# Function to display overlay and select cropping area
def select_crop_area(overlay_image):
    global top_left, bottom_right
    top_left, bottom_right = None, None

    fig, ax = plt.subplots(1)
    ax.imshow(overlay_image, cmap='gray')
    ax.set_title("Select top-left and bottom-right corners for cropping")
    rect_selector = RectangleSelector(ax, line_select_callback,
                                      drawtype='box', useblit=True,
                                      button=[1], minspanx=5, minspany=5,
                                      spancoords='pixels', interactive=True)
    plt.show()
    return top_left, bottom_right

# Function to perform registration
def perform_registration(fixed_image, moving_image, transform_type="affine", bspline_grid_size=[4, 4]):
    # Temporarily convert to float32 for registration
    fixed_image_float32 = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image_float32 = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Initialize the transform
    if transform_type == "affine":
        initial_transform = sitk.CenteredTransformInitializer(fixed_image_float32,
                                                              moving_image_float32,
                                                              sitk.AffineTransform(2),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    elif transform_type == "bspline":
        initial_transform = sitk.BSplineTransformInitializer(fixed_image_float32, bspline_grid_size)

    # Set up the registration
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=80)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.01,
                                                                 minStep=1e-5,
                                                                 numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Add the iteration command to monitor the progress
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    # Perform the registration
    final_transform = registration_method.Execute(fixed_image_float32, moving_image_float32)

    return final_transform

# Perform registration and cropping for each image pair
for he_file in os.listdir(hema_path):
    if he_file.endswith('.tif'):
        # Get the corresponding DAPI, CD20, CD4, BCL2, IRF4, CD15, PAX5, PD1, and HE files based on the naming convention
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

        # Full paths to the images
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
        print('Reading images')
        fixed_image = sitk.ReadImage(hema_image_path, sitk.sitkUInt8)
        fixed_image = sitk.InvertIntensity(fixed_image, maximum=255)
        moving_image = sitk.ReadImage(dapi_image_path, sitk.sitkUInt8)
        he_image = sitk.ReadImage(he_image_path, sitk.sitkVectorUInt8)

        # Choose the type of registration (affine or bspline)
        transform_type = "affine"

        # Set the B-spline grid size (if using B-spline)
        bspline_grid_size = [4, 4]

        # Perform the registration
        print(f'Performing {transform_type} registration')
        final_transform = perform_registration(fixed_image, moving_image, transform_type=transform_type, bspline_grid_size=bspline_grid_size)

        # Apply the transformation to the DAPI, CD20, CD4, BCL2, IRF4, CD15, PAX5, and PD1 images
        print('Apply the transformation to the DAPI, CD20, CD4, BCL2, IRF4, CD15, PAX5, and PD1 images')
        registered_dapi_image_float32 = sitk.Resample(
            sitk.Cast(sitk.ReadImage(dapi_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
            fixed_image, final_transform, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)

        registered_cd20_image_float32 = sitk.Resample(
            sitk.Cast(sitk.ReadImage(cd20_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
            fixed_image, final_transform, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)

        registered_cd4_image_float32 = sitk.Resample(
            sitk.Cast(sitk.ReadImage(cd4_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
            fixed_image, final_transform, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)

        registered_bcl2_image_float32 = sitk.Resample(
            sitk.Cast(sitk.ReadImage(bcl2_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
            fixed_image, final_transform, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)

        registered_irf4_image_float32 = sitk.Resample(
            sitk.Cast(sitk.ReadImage(irf4_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
            fixed_image, final_transform, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)

        registered_cd15_image_float32 = sitk.Resample(
            sitk.Cast(sitk.ReadImage(cd15_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
            fixed_image, final_transform, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)

        registered_pax5_image_float32 = sitk.Resample(
            sitk.Cast(sitk.ReadImage(pax5_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
            fixed_image, final_transform, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)

        registered_pd1_image_float32 = sitk.Resample(
            sitk.Cast(sitk.ReadImage(pd1_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
            fixed_image, final_transform, sitk.sitkLinear, 0.0,
            sitk.sitkFloat32)

        # Convert back to uint8 after registration
        print('Convert back to uint8')
        registered_dapi_image = sitk.Cast(sitk.RescaleIntensity(registered_dapi_image_float32, 0, 255), sitk.sitkUInt8)
        registered_cd20_image = sitk.Cast(sitk.RescaleIntensity(registered_cd20_image_float32, 0, 255), sitk.sitkUInt8)
        registered_cd4_image = sitk.Cast(sitk.RescaleIntensity(registered_cd4_image_float32, 0, 255), sitk.sitkUInt8)
        registered_bcl2_image = sitk.Cast(sitk.RescaleIntensity(registered_bcl2_image_float32, 0, 255), sitk.sitkUInt8)
        registered_irf4_image = sitk.Cast(sitk.RescaleIntensity(registered_irf4_image_float32, 0, 255), sitk.sitkUInt8)
        registered_cd15_image = sitk.Cast(sitk.RescaleIntensity(registered_cd15_image_float32, 0, 255), sitk.sitkUInt8)
        registered_pax5_image = sitk.Cast(sitk.RescaleIntensity(registered_pax5_image_float32, 0, 255), sitk.sitkUInt8)
        registered_pd1_image = sitk.Cast(sitk.RescaleIntensity(registered_pd1_image_float32, 0, 255), sitk.sitkUInt8)

        # Save the registered images
        print('Saving the final cropped images')
        sitk.WriteImage(he_image, os.path.join(he_output_path, f"final_{he_file}"))
        sitk.WriteImage(registered_dapi_image, os.path.join(dapi_output_path, f"{dapi_file}"))
        sitk.WriteImage(registered_cd20_image, os.path.join(cd20_output_path, f"{cd20_file}"))
        sitk.WriteImage(registered_cd4_image, os.path.join(cd4_output_path, f"{cd4_file}"))
        sitk.WriteImage(registered_bcl2_image, os.path.join(bcl2_output_path, f"{bcl2_file}"))
        sitk.WriteImage(registered_irf4_image, os.path.join(irf4_output_path, f"{irf4_file}"))
        sitk.WriteImage(registered_cd15_image, os.path.join(cd15_output_path, f"{cd15_file}"))
        sitk.WriteImage(registered_pax5_image, os.path.join(pax5_output_path, f"{pax5_file}"))
        sitk.WriteImage(registered_pd1_image, os.path.join(pd1_output_path, f"{pd1_file}"))

print("Processing complete.")
