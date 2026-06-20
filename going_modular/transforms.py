from torchvision import transforms

def create_data_transforms(
    size,
    augmentation=False,
    aug_scale=31,
    normalise=False,
    mean=None,
    std=None):

    """
    This helper function designs transformations that can be used to
    increase the variety in our data

    Args:
      size: A tuple of what the width and height of our images are
      augmentation: Whether our images should be augmented
      aug_scale: The scale of our augmentation

    Returns:
      transforms.Compose(transform_list): The transformation we desire
    """


    transform_list = [
        transforms.Resize(size)
    ]

    if augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.TrivialAugmentWide(
                num_magnitude_bins=aug_scale
            )
        ])

    transform_list.append(transforms.ToTensor())

    if normalise:
        transform_list.append(
            transforms.Normalize(mean, std)
        )

    return transforms.Compose(transform_list)