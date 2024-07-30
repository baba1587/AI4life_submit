"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm | gzip -c > example-algorithm.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path

import numpy as np
import torch
import SimpleITK
import tifffile

# Name of the expected input and output folders. CHANGE depending on the dataset.

# unstructured noise
# INPUT_PATH = Path("/input/images/image-stack-unstructured-noise/")
# OUTPUT_PATH = Path("/output/images/image-stack-denoised/")
# MODEL_PATH = Path("resources/model.pth")

# sturucted noise
INPUT_PATH = Path("/input/images/image-stack-structured-noise/")
OUTPUT_PATH = Path("/output/images/image-stack-denoised/")
# # # Path to the resource containing YOUR model. See 'src/create_model.py' for an example.
MODEL_PATH = Path("resources/model.pth")

# test
# INPUT_PATH = Path("/Users/shantong/github/grandChallenge/AI4Life/code/sunshine1/")
# OUTPUT_PATH = Path("/Users/shantong/github/grandChallenge/AI4Life/code/AI4Life-MDC24-example-submission-main/test/output/images/image-stack-structured-noise/")

# MODEL_PATH = Path("/Users/shantong/github/grandChallenge/AI4Life/code/AI4Life-MDC24-example-submission-main/resources/model.pth")


def show_torch_cuda_info():
    """Print cuda information, so it can be availiable in the container logs"""
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(
        f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(
            f"\tcurrent device: { (current_device := torch.cuda.current_device())}"
        )
        print(
            f"\tproperties: {torch.cuda.get_device_properties(current_device)}"
        )
    print("=+=" * 10)
    print("\n")


def save_result_image_mha(image_array: np.ndarray, result_path: Path):
    """Save the result denoised image.
    Be careful to save results only in .mha format!
    Otherwise, the container will fail"""
    print(f"Writing image to: {result_path}")
    mha_image = SimpleITK.GetImageFromArray(image_array)
    SimpleITK.WriteImage(mha_image, result_path, useCompression=True)


def save_result_image_tiff(image_array: np.ndarray, result_path: Path):
    print(f"Writing an image to: {result_path}")
    with tifffile.TiffWriter(result_path) as out:
        out.write(image_array, resolutionunit=2)


def normalize(image, min_val=0.0, max_val=1.0):

    # if isinstance(image, np.ndarray):
    # image = np.array(image, dtype=np.float32)
    # image = torch.tensor(image, dtype=torch.float32)

    image_min = float(torch.min(image))
    image_max = float(torch.max(image))

    normalized_image = (image - image_min) / (image_max - image_min)
    normalized_image = normalized_image * (max_val - min_val) + min_val
    return normalized_image, image_min, image_max


def unnormalize(image, image_min, image_max):

    unnormalized_image = image * (image_max - image_min) + image_min
    return unnormalized_image


def read_image(image_path: Path) -> (torch.Tensor, np.ndarray):
    """Read input noisy image from image_path"""
    print(f"Reading image: {image_path}")
    input_array = tifffile.imread(image_path)
    input_array = input_array.astype(np.float32)
    print(f"Loaded image shape: {input_array.shape}")
    original_shape = input_array.shape
    # For this example, we will flatten the samples and channels to predict images one by one
    # input_array = input_array.reshape(
    #     (-1, input_array.shape[-2], input_array.shape[-1])
    # )

    # input_array,original_min, original_max = normalize(input_array)

    input_tensor = torch.from_numpy(input_array)
    print(f"Final input shape: {input_tensor.shape}")
    return input_tensor, original_shape  #


def main():
    show_torch_cuda_info()

    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    input_files = sorted(INPUT_PATH.glob(f"*.tif*"))
    print(f"Found files: {input_files}")

    print(f"Loading model: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(MODEL_PATH).to(device)
    model.eval()

    for input_file in input_files:
        input_tensor, original_shape = read_image(input_file)
        input_tensor = input_tensor.to(device)

        print("Running inference...")
        result = np.zeros_like(input_tensor.cpu().numpy(), dtype=np.float32)

        # unstructed noise
        # Process each image independently
        # for i in range(input_tensor.shape[0]):
        #     for j in range(input_tensor.shape[1]):
        #         # print("input_tensor shape : ",input_tensor[0,:,:].shape)
        #         single_image_tensor = input_tensor[i, j, :, :].unsqueeze(0).unsqueeze(0).to(device)
        #         single_image_tensor, original_min, original_max = normalize(single_image_tensor)
        #         with torch.no_grad():
        #             denoised_image = model(single_image_tensor).squeeze().cpu().numpy()
        #             denoised_image = unnormalize(denoised_image, original_min, original_max)

        #         # import ipdb
        #         # ipdb.set_trace()

        #         print("denoised image shape : ",denoised_image.shape)
        #         result[i, j, :, :] = denoised_image

        # structed noise
        for i in range(input_tensor.shape[0]):
            # print("input_tensor shape : ",input_tensor[0,:,:].shape)
            single_image_tensor = input_tensor[i, :, :].unsqueeze(0).unsqueeze(
                0).to(device)
            single_image_tensor, original_min, original_max = normalize(
                single_image_tensor)
            with torch.no_grad():
                denoised_image = model(
                    single_image_tensor).squeeze().cpu().numpy()
                denoised_image = unnormalize(denoised_image, original_min,
                                             original_max)
            print("denoised image shape : ", denoised_image.shape)
            result[i, :, :] = denoised_image

        print(f"Output shape: {result.shape}")

        output_path = OUTPUT_PATH / f"{input_file.stem}.tif"
        save_result_image_tiff(image_array=result, result_path=output_path)


if __name__ == "__main__":
    main()
