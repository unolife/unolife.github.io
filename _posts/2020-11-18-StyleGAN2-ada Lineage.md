---
title: "StyleGAN2-ada 계보(lineage)"
date: 2020-11-18 14:10:00 -0400
categories: GAN
---
StyleGAN2-ada 코드를 돌려보다가, 해당 논문의 backbone을 거슬러가면서 공부할 필요성을 느끼고<br> 논문과 코드를 공부한 내용을 기록하기위해 만든 첫번째 포스트

2019.12.3 - styleGAN / <a href="https://github.com/NVlabs/stylegan">Github</a> / <a href="https://arxiv.org/pdf/1912.04958v1">Paper</a><br>
2020.03.23 - styleGAN2 / <a href="https://github.com/NVlabs/stylegan2">Github</a> / <a href="https://arxiv.org/pdf/1912.04958v2">Paper</a><br>
2020.10.07 - styleGAN-ada / <a href="https://github.com/NVlabs/stylegan2-ada">Github</a> / <a href="https://arxiv.org/pdf/2006.06676">Paper</a><br>

- <b>styleGAN2-ada</b><br>
  styleGAN2-ada 논문은 "Training Generative Adversarial Networks with Limited Data"라는 이름으로 발표됐다.<br>
  styleGAN2를 기반으로 소량의 데이터셋에서도 학습이 가능하도록 만든 논문이다.<br>
  <a href="https://arxiv.org/pdf/2006.06676v1">version1</a>(2020.06.11)도 있는데,v1 기반으로 v2에 추가적인 연구 성과가 더해진것이기 때문에 v2만 읽어도 될거 같다. v2에 v1의 내용이 대부분 들어 있다.

