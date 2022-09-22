Week 3 과제

도전한 내용: 실습이 PyTorch 기반으로 되어있어서, 내가 자주 사용한 Tensorflow 기반으로 Flower를 사용해보려고 함.

과정: Server를 그대로 두고, PyTorch 기반인 코드를 Tensorflow 기반으로 바꾸었음.
https://flower.dev/docs/quickstart-tensorflow.html 페이지를 참고함.

결과: 연합학습 실패. Input Shape가 맞지 않은 문제도 있었고, Memory 오버 문제도 보고.

실패 원인: PyTorch와 Tensorflow의 구조는 너무나도 많이 달라서 포팅하는 데 문제가 생긴 것으로 보고 있음. 또한 GPU를 한꺼번에 돌리다보니까 GPU의 VRAM이 고갈되는 현상이 나온 것으로 추측.
그러면 왜 하려고 했는가?: Tensorflow 기반의 연합학습 코드가 상당히 간결하였기에 활용하기에 좋아보여서 했음.

향후 계획: CNN 부분 고치고 다시 실행해볼 예정.
또한 localhost 이외에도 공인 cloud 환경에서도 돌릴 수 있도록 할 예정.
