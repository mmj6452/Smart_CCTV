
import tensorflow
import numpy as np
import cv2
from function.AI_CCTV import AI_processing
from function.AI_CCTV import draw_warning
from function.AI_CCTV import Layout_resize
from function.AI_CCTV import image_sum
from function.AI_CCTV import make_layout
from function.AI_CCTV import histogram_equalization

# 모델 주소 저장
model_address ='Smart_CCTV\Model\keras_model.h5'
# 모델값 가져오기
model = tensorflow.keras.models.load_model(model_address)

# 카메라에서 비디오 읽어오기
webcam = cv2.VideoCapture(0)
if webcam.isOpened() == False:
    raise Exception("카메라 연결이 안됩니다.")

# 읽어올 프레임을 qHD로 맞춰주기
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
webcam.set(cv2.CAP_PROP_ZOOM,1)
webcam.set(cv2.CAP_PROP_FOCUS,0)

#비디오 writer 구성
size = (round(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)),round(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = 4
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('Smart_CCTV/video/output.avi', fourcc, fps, size)
if out.isOpened() == False: 
    raise Exception("비디오 쓰기 불가")
    webcam.release()
    sys.exit()
    
#Latout 이미지 가져오기
main_clicked = Layout_resize(cv2.imread("Smart_CCTV/Layout/main_clicked.png"))
main_unclicked = Layout_resize(cv2.imread("Smart_CCTV/Layout/main_unclicked.png"))
edge_clicked = Layout_resize(cv2.imread("Smart_CCTV/Layout/edge_clicked.png"))
edge_unclicked = Layout_resize(cv2.imread("Smart_CCTV/Layout/edge_unclicked.png"))
improved_clicked = Layout_resize(cv2.imread("Smart_CCTV/Layout/improved_clicked.png"))
improved_unclicked = Layout_resize(cv2.imread("Smart_CCTV/Layout/improved_unclicked.png"))
original_clicked = Layout_resize(cv2.imread("Smart_CCTV/Layout/original_clicked.png"))
original_unclicked = Layout_resize(cv2.imread("Smart_CCTV/Layout/original_unclicked.png"))

#변수 선언 및 초기화
human_detect = False
frame_cnt = 0
frame_num = 0
Layout_state = 0
#bool 값들을 더해서 두개이상의 조건이 감지되면 빠르게 반응
detection_weight = human_detect

#무한반복
while True:
    #카메라에서 프레임가져오기 ret은 bool값으로 프레임이 가져와지면 True
    ret, frame = webcam.read()
    if not ret: break
    save = frame
    frame_num = frame_num + 1
    frame = cv2.resize(frame, (1280,960), interpolation=cv2.INTER_AREA)

    print("사람이 감지된 프레임갯수",frame_cnt)
    detection_weight = human_detect
    
    #키입력을 받는다
    #d를 입력하면 break 오른쪽 화살표:Layout_state를 +1해준다 왼쪽은 반대
    Key = cv2.waitKeyEx(100)
    if Key == 100: 
        break
    if Key == 2555904:
        Layout_state = Layout_state + 1
        if Layout_state > 3:
            Layout_state = 0
           
    if Key == 2424832:
        Layout_state = Layout_state - 1
        if Layout_state < 0:
            Layout_state = 3
            
    
    #읽어온 프레임을 AI처리해서 0 ~ 1의 사이의 사람일 확률과 아닐 확률이 나온다.
    Image_prediction = AI_processing(frame,model)

    #레이아웃 만드는 부분을 함수로 제작했지만 작동이 안되는 문제가 생겨서 바꿈
    if (Layout_state == 0):
        S_img_1 = histogram_equalization(frame)
        S_img_2 = cv2.Canny(frame,100,255)
        frame = image_sum(frame,S_img_1,S_img_2)
        frame[20:84,752:864] =  main_clicked
        frame[20:84,884:996] =  original_unclicked
        frame[20:84,1016:1128] =  improved_unclicked
        frame[20:84,1148:1260] =  edge_unclicked
        
    elif (Layout_state == 1):
        frame[20:84,752:864] =  main_unclicked
        frame[20:84,884:996] =  original_clicked
        frame[20:84,1016:1128] =  improved_unclicked
        frame[20:84,1148:1260] =  edge_unclicked
        
    elif (Layout_state == 2):
        frame = histogram_equalization(frame)
        #sharpening(frame)
        frame[20:84,752:864] =  main_unclicked
        frame[20:84,884:996] =  original_unclicked
        frame[20:84,1016:1128] =  improved_clicked 
        frame[20:84,1148:1260] =  edge_unclicked
    elif (Layout_state == 3):
        #Canny기법으로 엣지를 찾아낸다
        frame = cv2.Canny(frame,100,255)
        #그레이스케일 이미지를 BGR형태로 반환한다.
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        frame[20:84,752:864] =  main_unclicked
        frame[20:84,884:996] =  original_unclicked
        frame[20:84,1016:1128] =  improved_unclicked
        frame[20:84,1148:1260] =  edge_clicked
    
    
    #이미지에 사람이 없을 확률이 더 높은므로 감지 변수를 초기화해준다
    if (Image_prediction[0,0] < Image_prediction[0,1]):
        print('Non')
        cv2.putText(frame, 'non', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        frame_cnt = 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        human_detect = False

    #이미지에 사람이 있을 확률이 더 높으므로 프레임카운트에 +1을 해주고 human_detect를 true로 만들어준다.
    else:
        cv2.putText(frame, 'Detect', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        print('Detect')
        human_detect = True
        frame_cnt = frame_cnt + detection_weight
        
        path = 'Smart_CCTV/CCTV_img/'+str(frame_num)+'.jpg'
        print("해당 프레임이 저장되었습니다."+str(path))
        cv2.imwrite(str(path),save)
        out.write(save)
        
    #10개의 프레임이 지날때까지 사람이 계속 있을 경우    
    if (frame_cnt >= 10):
        #짝수 프레임일때 붉은색으로 경고를 해준다.
        if(frame_cnt % 2 == 0):
            draw_warning(frame)
            
    cv2.imshow("VideoFrame", frame)
out.release()
webcam.release()