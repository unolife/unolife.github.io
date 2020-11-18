---
title: "StyleGAN2-ada 계보(lineage)"
date: 2020-11-18 14:10:00 -0400
categories: GAN
---
styleGAN2-ada 코드를 돌려보다가, 논문을 거슬러가면서 공부할 필요성을 느끼고 논문과 코드를 공부한 내용을 기록하기위해 만든 첫번째 포스트

2014.06.10 - GAN / <a href="https://arxiv.org/pdf/1406.2661.pdf">Paper </a><br>
2017.10.27 - PGGAN (baseline of styleGAN) / <a href="https://arxiv.org/pdf/1710.10196v1.pdf">Paper v1</a> / <a href="https://arxiv.org/pdf/1710.10196v3.pdf">Paper v3 </a> (2018.02.26)<br>
2018.12.12 - styleGAN (First Version) / <a href="https://arxiv.org/pdf/1812.04948v1.pdf">Paper v1</a> /<a href="https://arxiv.org/pdf/1812.04948v3.pdf">Paper v3</a> (2019.03.29)<br>
2019.12.3 - styleGAN (Analyzing and Improving the Image Quality of StyleGAN) / <a href="https://github.com/NVlabs/stylegan">Github</a> / <a href="https://arxiv.org/pdf/1912.04958v1.pdf">Paper</a><br>
2020.03.23 - styleGAN2 / <a href="https://github.com/NVlabs/stylegan2">Github</a> / <a href="https://arxiv.org/pdf/1912.04958v2.pdf">Paper</a><br>
2020.10.07 - styleGAN2-ada / <a href="https://github.com/NVlabs/stylegan2-ada">Github</a> / <a href="https://arxiv.org/pdf/2006.06676.pdf">Paper</a><br>

- GAN(Generative Adversarial Networks)
  GAN에 대한 정보는 무수히 많기 때문에 <a href="https://www.tensorflow.org/tutorials/generative/dcgan?hl=ko">tensorflow 공식 문서</a>에 있는 Generator와 Discriminator 두 모델의 Keras layer들만 살펴보고 넘어갈거다.
  - Generator
    ```python
    def make_generator_model():
      model = tf.keras.Sequential()
      model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Reshape((7, 7, 256)))
      assert model.output_shape == (None, 7, 7, 256) # 주목: 배치사이즈로 None이 주어집니다.

      model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
      assert model.output_shape == (None, 7, 7, 128)
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      assert model.output_shape == (None, 14, 14, 64)
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
      assert model.output_shape == (None, 28, 28, 1)

      return model
    ```
    > Dense(7*7*256, use_bias=False, input_shape=(100,)))<br>
      input arrays of shape = (None, 100)<br>
      output arrays of shape = (None, 12544)

    > Conv2DTranspose<br>
      Input:<br>
      Output:
    
    
  - Discriminator
    ```python
    def make_discriminator_model():
      model = tf.keras.Sequential()
      model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                       input_shape=[28, 28, 1]))
      model.add(layers.LeakyReLU())
      model.add(layers.Dropout(0.3))

      model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
      model.add(layers.LeakyReLU())
      model.add(layers.Dropout(0.3))

      model.add(layers.Flatten())
      model.add(layers.Dense(1))

      return model
    ```
   
   
   - Generator 학습 방식(<a href="https://developers.google.com/machine-learning/gan/generator">참조</a>)<br>
     So we train the generator with the following procedure:
    1. Sample random noise.
    1. Produce generator output from sampled random noise.
    1. Get discriminator "Real" or "Fake" classification for generator output.
    1. Calculate loss from discriminator classification.
    1. Backpropagate through both the discriminator and generator to obtain gradients.
    1. Use gradients to change only the generator weights.<br>
    This is one iteration of generator training. <br>
    <br>랜덤 노이즈로부터 이미지를 만들고, D 모델에 넣어서 결과(real/fake)를 얻고, D 분류 결과에 대한 loss를 계산한다.<br>
    G와 D 모두 역전파를 해서 얻은 gradient(경사, 기울기)를 이용해서 G의 w(가중치)들을 갱신한다.

- <b>styleGAN2-ada</b><br>
  styleGAN2-ada 논문은 "Training Generative Adversarial Networks with Limited Data"라는 이름으로 발표됐다.<br>
  styleGAN2를 기반으로 소량의 데이터셋에서도 학습이 가능하도록 만든 논문이다.<br>
  <a href="https://arxiv.org/pdf/2006.06676v1.pdf">version1</a>(2020.06.11)도 있는데,v1 기반으로 v2에 추가적인 연구 성과가 더해진것이기 때문에 v2만 읽어도 될거 같다. v2에 v1의 내용이 대부분 들어 있다.

cf) visualization for model: https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
