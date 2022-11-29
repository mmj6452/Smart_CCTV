# Smart_CCTV
 
IoT영상통신시스템의 기말고사 대체 프로젝트

사전설치해야할 라이브러리

<pre><code>
    pip install opencv-python
    pip install tensorflow
    </code></pre>



폴더 구조
CCTV_img폴터는 사람이 감지되었을때 프레임들을 전부 이미지로써 저장하는 파일이다
video폴더는 사람이 감지되었을때 프레임들을 영상으로 저장하는 파일이다.
function폴더는 해당 프로그램을 구현함으로써 필요한 함수들을 구현하여 AI_CCTV라는 파일로 저장해두었다
Layout폴더는 Layout 이미지를 모아서 png형태로 저장해두고 docx파일로도 모아두었다.
Model폴더에는 Roboflow에서 받아온 사람이미지 데이터셋을 이용하여 티쳐블머신에 돌려만든 카라스 모델이 저장되어있다.


image processing task는 구현한 함수를 통해 실질적으로 작동하는 코드이다.
모델읽어오기
카메라에서 비디오 읽어오기
프레임을 비디오 또는 이미지로 저장하기
프레임을 읽어와서 사람이 감지되었는지 확인하기
감지가 되면 경고해주기
영상을 엣지,히스토그램평활화등의 방법으로 전처리하기
레이아웃 구성하기
등의 기능들이 구현되어있다.

AI_CCTV는 위의 image processing task를 구현하기 위해 필요한 함수들을 모아놓은 파일이다.
모델을 이용하여 사람이 감지되었는지 확인하는 함수
<pre><code>AI_processing(frame,model)</code></pre>
frame에는 이미지를 넣어주며
model에는 keras모델파일의 경로를 지정해준다
해당하는 오브젝트가 존재할 확률를 배열로 반환해준다.

프레임에 경고해주는 붉은 사각형을 그리는 함수
<pre><code>draw_warning(frame)</code></pre>
frame에 이미지를 넣어준다.
해당이미지의 테두리에 붉은색 테두리를 그려서 이미지를 반환해준다.

레이아웃 사이즈를 조절하는 함수
<pre><code>Layout_resize(img)</code></pre>
img에 이미지를 넣어주면
해당이미지를 112*64의 크기로 반환해준다.

필터로 컨볼루션을 진행해주는 함수
<pre><code>
    filter(image , mask)
    filter2(image , mask)
</code></pre>
image에 이미지를 넣어준다
mask에 mask를 넣어준다.
이미지에 마스크를 이용하여 컨볼루션을 한후 이미지값을 반환한다.

레이아웃을 만들어주는 함수
<pre><code>make_layout(frame,Layout_state,original_clicked,original_unclicked,edge_clicked,edge_unclicked,improved_clicked,improved_unclicked)</code></pre>
프레임에 이미지
그후 112*64크기에 각각맞는 버튼에 맞는 이미지를 넣어주면된다.
frame 오른쪽위에 버튼 3개를 생성해준다.
해당 함수가 함수로 작성해주면 작동안되는 문제가 발생하여 실제코드에는 사용하지 않고 직접 넣어줬다

밝기값에 대해 히스토그램이퀄라이제이션을 진행해주는 함수
<pre><code>histogram_equalization(frame)</code></pre>
frame에 이미지를 넣어준다
밝기값에 대해서만 이퀄라이제이션을 진행한 이미지를 반환해준다.

이미지 샤프닝을 해주는 함수
<pre><code>sharpening(frame)</code></pre>
image에 이미지를 넣어준다
이미지에 대해 샤프닝을 진행한 이미지를 반환해준다.

3개의 이미지를 한화면에 보여주기 위해 합쳐주는 함수
<pre><code>image_sum(B_img,S_img_1,S_img_2)</code></pre>
B_img 좌측에 띄어주고 싶은 이미지를 넣어준다 (이미지의 크기문제로 절반이 잘린형태로 반환이 된다.)
S_img_1 오른쪽 위쪽에 띄어주고 싶은 이미지를 넣어준다.
S_img_2 왼쪽 위쪽에 띄어주고 싶은 이미지를 넣어준다.
3개의 이미지를 합쳐서 하나의 화면에 나타내 준다.


/////////////////////////////////////////////////////////

버그났을때 대처법 경우 

경로에 이상이 있을 경우 처음 프로젝트를 열때 파이참에서 파이참 프로젝트 열기를 smart CCTV폴더로 지정해줄것
