<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# A way to compress the XLM cross language NMT model

## Knowledge Distillation

 Knowledge Distillation is a way to learn the output of a big model, it was proofed very useful in learn Big model's output to improve the accuracy of the Small model, the most important part of it is using knowledge of Softmax.

In general, we could regard the neural networks as lot of matrix that save the information of the network. So, for a neural network, how do we get the information of it, Knowledge Distillation is a way to get the information of the matrix, we first explain the standard softmax function:

![{\displaystyle \sigma (\mathbf {z} )_{i}={\frac {e^{z_{i))}{\sum _{j=1}^{K}e^{z_{j)))){\text{ for ))i=1,\dotsc ,K{\text{ and ))\mathbf {z} =(z_{1},\dotsc ,z_{K})\in \mathbb {R} ^{K))](https://wikimedia.org/api/rest_v1/media/math/render/svg/bdc1f8eaa8064d15893f1ba6426f20ff8e7149c5)

it was a kind of Normalized function, In above example, it has $K$ kinds of classifications, and the result is for class $J$, then $\sigma(z)_j$ is biggest, and it was much bigger than others that $i\neq j$ , In this way, we can only get information about the classification $J$ but others,  

To consider another way, we first smooth the matrix's output $z_{i}$ , 