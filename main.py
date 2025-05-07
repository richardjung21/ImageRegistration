import cv2
import numpy as np
from skimage import registration
import SimpleITK as sitk
import matplotlib.pyplot as plt
from dataset.oai import OAI
import torch

def feature_based_registration(img1, img2):
    """Feature-based registration using SIFT features."""
    # Convert torch tensors to numpy arrays
    img1_np = img1.squeeze().permute(1, 2, 0).numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).numpy()
    
    # Convert to grayscale
    img1_gray = cv2.cvtColor((img1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("Not enough keypoints found in one or both images")
        return img1_gray  # Return original image if registration fails
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception as e:
        print(f"Error in matching: {str(e)}")
        return img1_gray
    
    # Store good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        print(f"Not enough good matches found: {len(good_matches)}")
        return img1_gray
    
    # Get matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Failed to compute homography matrix")
        return img1_gray
    
    # Warp image
    h, w = img1_gray.shape
    try:
        registered_img = cv2.warpPerspective(img1_gray, H, (w, h))
        return registered_img
    except Exception as e:
        print(f"Error in warping: {str(e)}")
        return img1_gray

def intensity_based_registration(img1, img2):
    """Intensity-based registration using phase correlation."""
    # Convert torch tensors to numpy arrays
    img1_np = img1.squeeze().permute(1, 2, 0).numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).numpy()
    
    # Convert to grayscale
    img1_gray = cv2.cvtColor((img1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Convert to float32
    img1_float = img1_gray.astype(np.float32)
    img2_float = img2_gray.astype(np.float32)
    
    # Calculate phase correlation
    shift, error, diffphase = registration.phase_cross_correlation(img1_float, img2_float)
    
    # Apply shift
    registered_img = np.roll(img1_gray, int(shift[0]), axis=0)
    registered_img = np.roll(registered_img, int(shift[1]), axis=1)
    
    return registered_img

def simpleitk_registration(img1, img2):
    """Registration using SimpleITK's registration framework."""
    # Convert torch tensors to numpy arrays
    img1_np = img1.squeeze().permute(1, 2, 0).numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).numpy()
    
    # Convert to grayscale
    img1_gray = cv2.cvtColor((img1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Convert to float32 for SimpleITK
    img1_gray = img1_gray.astype(np.float32)
    img2_gray = img2_gray.astype(np.float32)
    
    # Convert to SimpleITK images
    sitk_img1 = sitk.GetImageFromArray(img1_gray)
    sitk_img2 = sitk.GetImageFromArray(img2_gray)
    
    # Initialize registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set similarity metric
    registration_method.SetMetricAsMeanSquares()
    
    # Set optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                    numberOfIterations=100)
    
    # Set initial transform
    initial_transform = sitk.CenteredTransformInitializer(sitk_img1, sitk_img2,
                                                        sitk.Euler2DTransform())
    registration_method.SetInitialTransform(initial_transform)
    
    try:
        # Execute registration
        final_transform = registration_method.Execute(sitk_img1, sitk_img2)
        
        # Apply transform
        registered_img = sitk.GetArrayFromImage(
            sitk.Resample(sitk_img1, sitk_img2, final_transform)
        )
        
        return registered_img
    except Exception as e:
        print(f"Error in SimpleITK registration: {str(e)}")
        return img1_gray

def mutual_information_registration(img1, img2):
    """Registration using mutual information metric."""
    # Convert torch tensors to numpy arrays
    img1_np = img1.squeeze().permute(1, 2, 0).numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).numpy()
    
    # Convert to grayscale
    img1_gray = cv2.cvtColor((img1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Convert to float32 for SimpleITK
    img1_gray = img1_gray.astype(np.float32)
    img2_gray = img2_gray.astype(np.float32)
    
    # Convert to SimpleITK images
    sitk_img1 = sitk.GetImageFromArray(img1_gray)
    sitk_img2 = sitk.GetImageFromArray(img2_gray)
    
    # Initialize registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set similarity metric to mutual information
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    # Set optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                    numberOfIterations=200,
                                                    convergenceMinimumValue=1e-6,
                                                    convergenceWindowSize=10)
    
    # Set initial transform
    initial_transform = sitk.CenteredTransformInitializer(sitk_img1, sitk_img2,
                                                        sitk.Euler2DTransform())
    registration_method.SetInitialTransform(initial_transform)
    
    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Set multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    
    try:
        # Execute registration
        final_transform = registration_method.Execute(sitk_img1, sitk_img2)
        
        # Get the final metric value
        final_metric = registration_method.GetMetricValue()
        print(f"Final metric value: {final_metric}")
        
        # Apply transform
        registered_img = sitk.GetArrayFromImage(
            sitk.Resample(sitk_img1, sitk_img2, final_transform)
        )
        
        return registered_img
    except Exception as e:
        print(f"Error in mutual information registration: {str(e)}")
        return img1_gray

def visualize_results(original1, original2, registered, method_name):
    """Visualize registration results."""
    # Convert torch tensors to numpy arrays
    original1_np = original1.squeeze().permute(1, 2, 0).numpy()
    original2_np = original2.squeeze().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(original1_np)
    plt.title('Original Image 1')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(original2_np)
    plt.title('Original Image 2')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(registered, cmap='gray')
    plt.title(f'Registered Image ({method_name})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize OAI dataset
    dataset = OAI(root_dir="/mnt/sdb3/OAI/dat/KL/Dataset", split='train')
    
    # Get a random pair of images
    img1, lbl1, img2, lbl2 = dataset.get_random_pair()
    
    print(f"Successfully loaded image pair with labels: {lbl1} and {lbl2}")
    
    # Perform different registration methods
    feature_registered = feature_based_registration(img1, img2)
    intensity_registered = intensity_based_registration(img1, img2)
    simpleitk_registered = simpleitk_registration(img1, img2)
    mi_registered = mutual_information_registration(img1, img2)
    
    # Visualize results
    visualize_results(img1, img2, feature_registered, "Feature-based")
    visualize_results(img1, img2, intensity_registered, "Intensity-based")
    visualize_results(img1, img2, simpleitk_registered, "SimpleITK")
    visualize_results(img1, img2, mi_registered, "Mutual Information")

if __name__ == "__main__":
    main()
