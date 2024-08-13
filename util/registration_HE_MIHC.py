import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Define the paths
hema_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_hema\uint8'
dapi_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\RGB_3channels\normalized_channel_1'
bcl2_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\RGB_3channels\normalized_channel_5'
pax5_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\RGB_3channels\normalized_channel_9'
he_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE'
he_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_HE\Registered_HE'
dapi_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_dapi'
bcl2_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_bcl2'
pax5_output_path = r'D:\Chang_files\workspace\Qupath_proj\hdk_codex\run6_mIHC\processed_images\registered_pax5'

# Create the output directories if they don't exist
os.makedirs(he_output_path, exist_ok=True)
os.makedirs(dapi_output_path, exist_ok=True)
os.makedirs(bcl2_output_path, exist_ok=True)
os.makedirs(pax5_output_path, exist_ok=True)

# Function to extract the common prefix from the filename
def extract_prefix(filename):
    return filename.split('.')[0].replace('hema_', '')

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
for hema_file in os.listdir(hema_path):
    if hema_file.endswith('.tif'):
        # Get the corresponding DAPI, BCL2, PAX5, and HE files based on the naming convention
        prefix = extract_prefix(hema_file)
        dapi_file = f"mIHC_{prefix}_channel_1.tif"
        bcl2_file = f"mIHC_{prefix}_channel_5.tif"
        pax5_file = f"mIHC_{prefix}_channel_9.tif"
        he_file = f"HE_{prefix}.tif"

        # Full paths to the images
        hema_image_path = os.path.join(hema_path, hema_file)
        dapi_image_path = os.path.join(dapi_path, dapi_file)
        bcl2_image_path = os.path.join(bcl2_path, bcl2_file)
        pax5_image_path = os.path.join(pax5_path, pax5_file)
        he_image_path = os.path.join(he_path, he_file)

        # Read the images as uint8
        print('Reading images')
        fixed_image = sitk.ReadImage(hema_image_path, sitk.sitkUInt8)
        moving_image = sitk.ReadImage(dapi_image_path, sitk.sitkUInt8)
        he_image = sitk.ReadImage(he_image_path, sitk.sitkVectorUInt8)

        # Temporarily convert to float32 for registration
        fixed_image_float32 = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image_float32 = sitk.Cast(moving_image, sitk.sitkFloat32)

        # Initialize the transform using AffineTransform for scaling, rotation, and translation
        print('Initialize the transform')
        initial_transform = sitk.CenteredTransformInitializer(fixed_image_float32,
                                                              moving_image_float32,
                                                              sitk.AffineTransform(2),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Set up the registration
        print('Set up the registration')
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=60)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.2,
                                                                     minStep=1e-6,
                                                                     numberOfIterations=800)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Perform the rigid registration
        print('Perform the registration')
        final_transform = registration_method.Execute(fixed_image_float32, moving_image_float32)

        # Apply the rigid transformation to the DAPI, BCL2, and PAX5 images
        print('Apply the transformation to the DAPI, BCL2, and PAX5 images')
        registered_dapi_image_float32 = sitk.Resample(moving_image_float32,
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        registered_bcl2_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(bcl2_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        registered_pax5_image_float32 = sitk.Resample(sitk.Cast(sitk.ReadImage(pax5_image_path, sitk.sitkUInt8), sitk.sitkFloat32),
                                                      fixed_image_float32, final_transform, sitk.sitkLinear, 0.0,
                                                      sitk.sitkFloat32)

        # Convert back to uint8 after rigid registration
        print('Convert back to uint8')
        registered_dapi_image = sitk.Cast(sitk.RescaleIntensity(registered_dapi_image_float32, 0, 255), sitk.sitkUInt8)
        registered_bcl2_image = sitk.Cast(sitk.RescaleIntensity(registered_bcl2_image_float32, 0, 255), sitk.sitkUInt8)
        registered_pax5_image = sitk.Cast(sitk.RescaleIntensity(registered_pax5_image_float32, 0, 255), sitk.sitkUInt8)

        # Overlay the registered DAPI with Hema image
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
            cropped_bcl2_image = crop_image_by_coordinates(registered_bcl2_image, top_left, bottom_right)
            cropped_pax5_image = crop_image_by_coordinates(registered_pax5_image, top_left, bottom_right)

            if apply_non_rigid == "yes":
                # Perform B-spline non-rigid registration on the cropped images
                print('Perform B-spline non-rigid registration')
                transformDomainMeshSize = [8] * cropped_hema_image.GetDimension()
                bspline_transform = sitk.BSplineTransformInitializer(cropped_hema_image,
                                                                     transformDomainMeshSize)

                registration_method.SetInitialTransform(bspline_transform, inPlace=False)
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=30)
                registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                                         numberOfIterations=200,
                                                         maximumNumberOfCorrections=5,
                                                         maximumNumberOfFunctionEvaluations=1000,
                                                         costFunctionConvergenceFactor=1e+7)
                registration_method.SetInterpolator(sitk.sitkLinear)

                bspline_transform = registration_method.Execute(sitk.Cast(cropped_hema_image, sitk.sitkFloat32),
                                                                sitk.Cast(cropped_dapi_image, sitk.sitkFloat32))

                # Apply the B-spline transformation to the cropped images
                print('Apply B-spline transformation to the cropped images')
                final_cropped_dapi_image = sitk.Resample(cropped_dapi_image,
                                                         cropped_hema_image,
                                                         bspline_transform,
                                                         sitk.sitkLinear, 0.0, cropped_dapi_image.GetPixelID())

                final_cropped_bcl2_image = sitk.Resample(cropped_bcl2_image,
                                                         cropped_hema_image,
                                                         bspline_transform,
                                                         sitk.sitkLinear, 0.0, cropped_bcl2_image.GetPixelID())

                final_cropped_pax5_image = sitk.Resample(cropped_pax5_image,
                                                         cropped_hema_image,
                                                         bspline_transform,
                                                         sitk.sitkLinear, 0.0, cropped_pax5_image.GetPixelID())

                # Second cropping after non-rigid registration
                print('Second cropping after non-rigid registration')
                overlay_image = 0.5 * sitk.GetArrayViewFromImage(cropped_hema_image) + 0.5 * sitk.GetArrayViewFromImage(final_cropped_dapi_image)
                top_left, bottom_right = select_crop_area(overlay_image)

                if top_left and bottom_right:
                    print('Final cropping')
                    cropped_he_image = crop_image_by_coordinates(cropped_he_image, top_left, bottom_right)
                    cropped_hema_image = crop_image_by_coordinates(cropped_hema_image, top_left, bottom_right)
                    final_cropped_dapi_image = crop_image_by_coordinates(final_cropped_dapi_image, top_left, bottom_right)
                    final_cropped_bcl2_image = crop_image_by_coordinates(final_cropped_bcl2_image, top_left, bottom_right)
                    final_cropped_pax5_image = crop_image_by_coordinates(final_cropped_pax5_image, top_left, bottom_right)

                    # Save the final cropped images
                    print('Saving the final cropped images')
                    sitk.WriteImage(cropped_he_image, os.path.join(he_output_path, f"final_{he_file}"))
                    sitk.WriteImage(cropped_hema_image, os.path.join(he_output_path, f"final_{hema_file}"))
                    sitk.WriteImage(final_cropped_dapi_image, os.path.join(dapi_output_path, f"final_{dapi_file}"))
                    sitk.WriteImage(final_cropped_bcl2_image, os.path.join(bcl2_output_path, f"final_{bcl2_file}"))
                    sitk.WriteImage(final_cropped_pax5_image, os.path.join(pax5_output_path, f"final_{pax5_file}"))
            else:
                # Save the cropped images after rigid registration without non-rigid registration
                print('Saving the cropped images after rigid registration')
                sitk.WriteImage(cropped_he_image, os.path.join(he_output_path, f"rigid_{he_file}"))
                sitk.WriteImage(cropped_hema_image, os.path.join(he_output_path, f"rigid_{hema_file}"))
                sitk.WriteImage(cropped_dapi_image, os.path.join(dapi_output_path, f"rigid_{dapi_file}"))
                sitk.WriteImage(cropped_bcl2_image, os.path.join(bcl2_output_path, f"rigid_{bcl2_file}"))
                sitk.WriteImage(cropped_pax5_image, os.path.join(pax5_output_path, f"rigid_{pax5_file}"))

        print(f"Finished processing {hema_file}")

print("Processing complete.")
