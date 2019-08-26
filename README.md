# Dizziness_tracker


>**딥러닝을 이용한 인물 분류기** 

<h2> 팀원</h2>
<ul>
  <li>김연우</li>
  <li>김효진</li>
  <li>윤원석</li>
</ul>

</hr>

<h2>현재까지의 진행사항</h2>

* **주제 선정**: <br>

  이유 :  기존 안진검사에 사용되는 frenzel 특수안경의 불편한점을 보완하고자, 동공추적을 이용한 평형기능 검사 시스템을 구축

* **개발 환경 구축을 위한 프로그램 설치** <br>
: open cv,pandas, dlib, cmake, numpy 등

* **코드 작성** <br>
camera open, object tracking, data processing, data augmentation 를 수행할 코드를 작성

* **데이터셋** <br>

<h3> https://drive.google.com/drive/u/2/folders/1OLAWywU4vC5e2MFPSaNRTVhl--alRwkp </h3>

(1) 동영상 데이터 셋 <br>
(2) 좌표값 데이터 셋 <br>

<h2>전체 계획대비 진행상황</h2> 

 **<진행 중 및 진행 완료>**
 * Object tracking을 python으로 구현하여 동공을 추적 완료 <br>
 * 데이터 전처리,data augmentation 완료 <br>
 * 동영상 1개당 약 350 frame으로 나누어 이미지 데이터를 CNN으로 학습 완료  <br>
 * 편리성과 접근성을 위하여 구현코드와 APP 연동 진행중<br>

 **<동공 분류기 수행 목표>**
 * 환자가 병원에 방문하지 않아도 자신의 증상에 대해 예측 가능(진단까지는 아님) <br>
 * 의사는 촬영한 눈의 움직임을 다시 분석해야 한다는 번거로움이 줄어듦 <br>
 * APP과 연동으로 편리성과 접근성 높힘<br>



<h2>참고자료</h2>

num| 사이트
--------- | ---------
1 | http://www.dizziness-and-balance.com/disorders/bppv/bppv-korean.htm
2 | http://www.ijircce.com/upload/2014/february/7J_A%20Survey.pdf
3 | http://www.ijsrp.org/research-paper-0513/ijsrp-p17106.pdf
4 | https://dgkim5360.tistory.com
5 | https://eehoeskrap.tistory.com/91
