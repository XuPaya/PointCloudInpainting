# Point Cloud Inpainting
This is a deep learning model for point cloud inpainting based on the classical PointNet model and a GAN architecture. The overall structure is similiar to [PCN](https://arxiv.org/abs/1808.00671), with a permutational invariance property. Therefore, the model runs extremly fast and include no spacial convolution or self attention.

## What is this project ##
This is a course project for the guaduate level course [COMP5214](https://course.cse.ust.hk/comp5214/project.html) in HKUST. No performance is guaranteed, and we expect no further update on this repo.

## Comparison with Existing methods ##
One should notice that this model has almost the same architecture as the [PCN](https://arxiv.org/abs/1808.00671) model. However, this project has also found the paper [Point Encoder GAN: A deep learning model for 3D point cloud inpainting](https://www.sciencedirect.com/science/article/pii/S0925231219317357) to have the exact same structure. All they did was simply change one FC layer to a so-called "deconv" layer, which has the same connectivity and same number of parameters as a ordinary FC-layer. **It is quite astounding that one can publish a paper by applying [PCN](https://arxiv.org/abs/1808.00671) on a much easier task than it was designed for, and claim that its result is novel and effective.**
