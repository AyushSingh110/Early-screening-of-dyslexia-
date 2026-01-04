import cv2

def split_into_patches(image, patch_size=224, stride=224):
    h, w, _ = image.shape
    patches = []

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)

    return patches
