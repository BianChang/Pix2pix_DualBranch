import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Define the paths
hema_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE\gray_HE'
dapi_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\channel_1_normalized'
cd20_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\channel_2_normalized'
cd4_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\channel_4_normalized'
bcl2_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\channel_5_normalized'
irf4_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\channel_6_normalized'
cd15_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\channel_7_normalized'
pax5_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\channel_9_normalized'
pd1_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\channel_10_normalized'
he_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE'
he_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE\Registered_HE'
dapi_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_dapi'
cd20_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_cd20'
cd4_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_cd4'
bcl2_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_bcl2'
irf4_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_irf4'
cd15_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_cd15'
pax5_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_pax5'
pd1_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_pd1'

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
    return filename.split('.')[0].replace('HE_', '')

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

# Ask the user whether to apply non-rigid registration
apply_non_rigid = input("Do you want to apply non-rigid registration? (yes/no): ").strip().lower()

# Perform registration and cropping for each image pair
for he_file in os.listdir(hema_path):
    if he_file.endswith('.tif'):
        print(he_file)
        # Get the corresponding DAPI, CD20, CD4, BCL2, IRF4, CD15, PAX5, PD1, and HE files based on the naming convention
        prefix = extract_prefix(he_file)
        dapi_file = f"mIHC_{prefix}_channel_1.tif"
        cd20_file = f"mIHC_{prefix}_channel_2.tif"
        cd4_file = f"mIHC_{prefix}_channel_4.tif"
        bcl2_file = f"mIHC_{prefix}_channel_5.tif"
        irf4_file = f"mIHC_{prefix}_channel_6.tif"
        cd15_file = f"mIHC_{prefix}_channel_7.tif"
        pax5_file = f"mIHC_{prefix}_channel_9.tif"
        pd1_file = f"mIHC_{prefix}_channel_10.tif"
        he_file = f"HE_{prefix}.tif"

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

        # Temporarily convert to float32 for registration
        fixed_image_float32 = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image_float32 = sitk.Cast(moving_image, sitk.sitkFloat32)

        # Initialize the transform using AffineTransform for scaling, rotation, and translation
        print('Initialize the transform')
        initial_transform = sitk.CenteredTransformInitializer(fixed_image_float32,
                                                              moving_image_float32,
                                                              sitk.Similarity2DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Set up the registration
        print('Set up the registration')
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=55)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.3,
                                                                     minStep=1e-4,
                                                                     numberOfIterations=30)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Add the iteration command to monitor the progress
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

        # Perform the rigid registration
        print('Perform the registration')
        final_transform = registration_method.Execute(fixed_image_float32, moving_image_float32)

        # Apply the rigid transformation to the DAPI, CD20, CD4, BCL2, IRF4, CD15, PAX5, and PD1 images
        print('Apply the transformation to the DAPI, CD20, CD4, BCL2, IRF4, CD15, PAX5, and PD1 images')
        registered_dapi_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(dapi_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        registered_cd20_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(cd20_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        registered_cd4_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(cd4_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                     fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                     sitk.sitkFloat32)

        registered_bcl2_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(bcl2_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        registered_irf4_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(irf4_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        registered_cd15_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(cd15_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        registered_pax5_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(pax5_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        registered_pd1_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(pd1_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                     fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                     sitk.sitkFloat32)

        # Convert back to uint8 after rigid registration
        print('Convert back to uint8')
        registered_dapi_image = sitk.Cast(sitk.RescaleIntensity(registered_dapi_image_float32, 0, 255), sitk.sitkUInt8)
        registered_cd20_image = sitk.Cast(sitk.RescaleIntensity(registered_cd20_image_float32, 0, 255), sitk.sitkUInt8)
        registered_cd4_image = sitk.Cast(sitk.RescaleIntensity(registered_cd4_image_float32, 0, 255), sitk.sitkUInt8)
        registered_bcl2_image = sitk.Cast(sitk.RescaleIntensity(registered_bcl2_image_float32, 0, 255), sitk.sitkUInt8)
        registered_irf4_image = sitk.Cast(sitk.RescaleIntensity(registered_irf4_image_float32, 0, 255), sitk.sitkUInt8)
        registered_cd15_image = sitk.Cast(sitk.RescaleIntensity(registered_cd15_image_float32, 0, 255), sitk.sitkUInt8)
        registered_pax5_image = sitk.Cast(sitk.RescaleIntensity(registered_pax5_image_float32, 0, 255), sitk.sitkUInt8)
        registered_pd1_image = sitk.Cast(sitk.RescaleIntensity(registered_pd1_image_float32, 0, 255), sitk.sitkUInt8)

        # Overlay the registered DAPI with fixed image
        print('Overlay the registered DAPI with Hema')
        overlay_image = 0.5 * sitk.GetArrayViewFromImage(fixed_image) + 0.5 * sitk.GetArrayViewFromImage(registered_dapi_image)

        # First cropping after rigid registration
        print('First cropping after rigid registration')
        top_left, bottom_right = select_crop_area(overlay_image)

        if top_left and bottom_right:
            # Crop images according to the selected region after rigid registration
            print('Cropping images after rigid registration')
            cropped_he_image = crop_image_by_coordinates(he_image, top_left, bottom_right)
            cropped_hema_image = crop_image_by_coordinates(fixed_image, top_left, bottom_right)
            cropped_dapi_image = crop_image_by_coordinates(registered_dapi_image, top_left, bottom_right)
            cropped_cd20_image = crop_image_by_coordinates(registered_cd20_image, top_left, bottom_right)
            cropped_cd4_image = crop_image_by_coordinates(registered_cd4_image, top_left, bottom_right)
            cropped_bcl2_image = crop_image_by_coordinates(registered_bcl2_image, top_left, bottom_right)
            cropped_irf4_image = crop_image_by_coordinates(registered_irf4_image, top_left, bottom_right)
            cropped_cd15_image = crop_image_by_coordinates(registered_cd15_image, top_left, bottom_right)
            cropped_pax5_image = crop_image_by_coordinates(registered_pax5_image, top_left, bottom_right)
            cropped_pd1_image = crop_image_by_coordinates(registered_pd1_image, top_left, bottom_right)

        if apply_non_rigid == "yes":

            registration_method_non_rigid = sitk.ImageRegistrationMethod()
            print('Perform Affine registration')

            # Initialize the transform using AffineTransform for scaling, rotation, translation, and shearing
            affine_transform = sitk.AffineTransform(cropped_hema_image.GetDimension())

            # Set the initial transform for affine registration
            registration_method_non_rigid.SetInitialTransform(affine_transform, inPlace=False)

            # Set the metric as Mattes Mutual Information
            registration_method_non_rigid.SetMetricAsMattesMutualInformation(numberOfHistogramBins=75)

            registration_method_non_rigid.SetOptimizerAsRegularStepGradientDescent(learningRate=0.01,
                                                                                       minStep=1e-6,
                                                                                       numberOfIterations=100)

            registration_method_non_rigid.SetInterpolator(sitk.sitkLinear)

            registration_method_non_rigid.AddCommand(sitk.sitkIterationEvent,
                                                    lambda: command_iteration(registration_method_non_rigid))

            bspline_transform = registration_method_non_rigid.Execute(
                sitk.Cast(cropped_hema_image, sitk.sitkFloat32),
                sitk.Cast(cropped_dapi_image, sitk.sitkFloat32))

            # affine
            print('Apply Affine transformation to the cropped images')
            final_cropped_dapi_image = sitk.Resample(cropped_dapi_image,
                                                    cropped_hema_image,
                                                    bspline_transform,
                                                    sitk.sitkLinear, 0.0, cropped_dapi_image.GetPixelID())

            final_cropped_cd20_image = sitk.Resample(cropped_cd20_image,
                                                    cropped_hema_image,
                                                    bspline_transform,
                                                    sitk.sitkLinear, 0.0, cropped_cd20_image.GetPixelID())

            final_cropped_cd4_image = sitk.Resample(cropped_cd4_image,
                                                   cropped_hema_image,
                                                   bspline_transform,
                                                   sitk.sitkLinear, 0.0, cropped_cd4_image.GetPixelID())

            final_cropped_bcl2_image = sitk.Resample(cropped_bcl2_image,
                                                    cropped_hema_image,
                                                    bspline_transform,
                                                    sitk.sitkLinear, 0.0, cropped_bcl2_image.GetPixelID())

            final_cropped_irf4_image = sitk.Resample(cropped_irf4_image,
                                                    cropped_hema_image,
                                                    bspline_transform,
                                                    sitk.sitkLinear, 0.0, cropped_irf4_image.GetPixelID())

            final_cropped_cd15_image = sitk.Resample(cropped_cd15_image,
                                                    cropped_hema_image,
                                                    bspline_transform,
                                                    sitk.sitkLinear, 0.0, cropped_cd15_image.GetPixelID())

            final_cropped_pax5_image = sitk.Resample(cropped_pax5_image,
                                                    cropped_hema_image,
                                                    bspline_transform,
                                                    sitk.sitkLinear, 0.0, cropped_pax5_image.GetPixelID())

            final_cropped_pd1_image = sitk.Resample(cropped_pd1_image,
                                                   cropped_hema_image,
                                                   bspline_transform,
                                                   sitk.sitkLinear, 0.0, cropped_pd1_image.GetPixelID())

            # Save the final cropped images
            print('Saving the final cropped images')
            sitk.WriteImage(cropped_he_image, os.path.join(he_output_path, f"final_{he_file}"))
            sitk.WriteImage(final_cropped_dapi_image, os.path.join(dapi_output_path, f"final_{dapi_file}"))
            sitk.WriteImage(final_cropped_cd20_image, os.path.join(cd20_output_path, f"final_{cd20_file}"))
            sitk.WriteImage(final_cropped_cd4_image, os.path.join(cd4_output_path, f"final_{cd4_file}"))
            sitk.WriteImage(final_cropped_bcl2_image, os.path.join(bcl2_output_path, f"final_{bcl2_file}"))
            sitk.WriteImage(final_cropped_irf4_image, os.path.join(irf4_output_path, f"final_{irf4_file}"))
            sitk.WriteImage(final_cropped_cd15_image, os.path.join(cd15_output_path, f"final_{cd15_file}"))
            sitk.WriteImage(final_cropped_pax5_image, os.path.join(pax5_output_path, f"final_{pax5_file}"))
            sitk.WriteImage(final_cropped_pd1_image, os.path.join(pd1_output_path, f"final_{pd1_file}"))

        else:
            # Save the cropped images after rigid registration without non-rigid registration
            print('Saving the cropped images after rigid registration')
            sitk.WriteImage(cropped_he_image, os.path.join(he_output_path, f"rigid_{he_file}"))
            sitk.WriteImage(cropped_dapi_image, os.path.join(dapi_output_path, f"rigid_{dapi_file}"))
            sitk.WriteImage(cropped_cd20_image, os.path.join(cd20_output_path, f"rigid_{cd20_file}"))
            sitk.WriteImage(cropped_cd4_image, os.path.join(cd4_output_path, f"rigid_{cd4_file}"))
            sitk.WriteImage(cropped_bcl2_image, os.path.join(bcl2_output_path, f"rigid_{bcl2_file}"))
            sitk.WriteImage(cropped_irf4_image, os.path.join(irf4_output_path, f"rigid_{irf4_file}"))
            sitk.WriteImage(cropped_cd15_image, os.path.join(cd15_output_path, f"rigid_{cd15_file}"))
            sitk.WriteImage(cropped_pax5_image, os.path.join(pax5_output_path, f"rigid_{pax5_file}"))
            sitk.WriteImage(cropped_pd1_image, os.path.join(pd1_output_path, f"rigid_{pd1_file}"))

        print(f"Finished processing {he_file}")

print("Processing complete.")
