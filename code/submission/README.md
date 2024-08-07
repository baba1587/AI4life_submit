<p align="center">
  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://rumc-gcorg-p-public.s3.amazonaws.com/b/756/denoising_21.x10.jpeg" width="100%">
  </a>
</p>

# Denoising challenge AI4Life-MDC24

-----

## Example submission container

Welcome to the AI4Life-MDC24 denoising challenge!
Check [The Challenge Page](https://ai4life-mdc24.grand-challenge.org/) for all the details about the challenge.

On this page, you can find an example submission for the Grand Challenge platform.

### Submission checklist

- [ ] Make sure that the `INPUT_PATH` and `OUTPUT_PATH` in your code are correct. More info [below](#data-location).
- [ ] Check that all the requirements are contained in the `requirements.txt` and `Dockerfile`.
- [ ] Test your container locally with `test_run.sh`. Check how much memory (RAM) your algorithm requires.
- [ ] Create a gzip archive from your image, run `docker save example-algorithm | gzip -c > example-algorithm.tar.gz`
- [ ] You are ready to submit! Go to [this page](https://ai4life-mdc24.grand-challenge.org/how-to-submit-you-results/) for further instructions on how to submit to the Grand Challenge.

### How to use this code

1. Example input image is stored in the `test/input/images/image-stack-structured-noise` folder.
2. Look through the contents of the [inference.py](inference.py) script.
3. Run [test_run.sh](test_run.sh) to build and test the container execution.
4. The resulting image should appear in the `test/output/images/image-stack-denoised` folder.

### What is in the example?

Here, we are showcasing an example pytorch model and its inference.
The model contains only a Gaussian Blur operator. The model is packaged into jit. See [create_model.py](create_model.py) for details.

We use a light python container with pytorch-cpu for this example, see [Dockerfile](Dockerfile). You can also use GPU version!

The container runs [inference.py](inference.py) script, which loops through the noisy images in the `INPUT_PATH` and applies the model to them individually.
The result denoised images are then saved into `OUTPUT_PATH` folder.

### Data location

#### For datasets containing structured noise

1. The input images in the container are stored as `/input/images/image-stack-structured-noise/<uuid>.tif`
2. The output of your algorithm should be saved as `/input/images/image-stack-denoised/<uuid>.tif`

#### For datasets containing unstructured noise

1. The input images in the container are stored as `/input/images/image-stack-unstructured-noise/<uuid>.tif`
2. The output of your algorithm should be saved as `/input/images/image-stack-denoised/<uuid>.tif`

For more details about the datasets, check out [Data description](https://ai4life-mdc24.grand-challenge.org/data-description/) page!

### Useful links

To learn more about Docker and how to write Dockerfile check out the [Official documentation](https://docs.docker.com/guides/get-started/).

Make sure to check Grand Challenge [documentation](https://grand-challenge.org/documentation/participate-in-a-challenge/) and [forum](https://grand-challenge.org/forums/forum/general-548/) with any questions you may have.  

For **any** other questions or issues, create a topic on the [challenge forum](https://grand-challenge.org/forums/forum/ai4life-microscopy-denoising-challenge-721/) or drop us an email through the *Email organizers* button on the challenge page.

#### Thank you for participating, and we are looking forward to receiving your submission
