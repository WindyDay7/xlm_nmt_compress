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

To consider another way, we first smooth the matrix's output $z_{i}$ ,  So that we can get more information about matrix, the we use the small model to learn the information to get the knowledge.

## Big changes on XLM

### Trainer

The big trainer could use the pretrain model by Facebook, it could get an very good result, then the small model comes from the big one. The way to get the smaller is cutting some layers of the big ones, this process could be implemented by Pytorch easily, you could reference my [blog](https://www.cnblogs.com/wevolf/p/12918217.html)  . It was not complex.

### Train step

While train knowledge distillation, two ways could be used, one is save the big models outputs, when train the smaller one, we cloud load it into memory, and use the output by smaller to calculate the output, another way could be used when you have a strong running environment, for me, I adopt this way, that is construct two model while train, we could  fit the output of the smaller to the big one, but we must consider how much should we fit to the bigger one. That is, in loss function, we need to consider the weight of learn from big model's output and the really output. 

## BIDGRU with BPE code

BIDGRU means Bi-direction GRU network, which means use bi-direction GRU to construct a Seq-to-Seq model, We know the XLM used the BPE code as the way to code the tokens, In this condition, I wonder what's performance of BIDGRU seq-to-seq model. And this part of strategy also need a big change in code, particularly in the construction of model. I specify this part of Code in Student_Model, which also will be train as Student_Model, but for now, I have some trouble in deal with batch_size in seq-to-seq model, the problem is that when should we stop the decoder step, and how to determine the end of the sentence.

