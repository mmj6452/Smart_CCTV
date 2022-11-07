import cv2
import numpy as np
import tensorflow


# 이미지 처리하기
def AI_processing(frame,model):
    # 모델링된 가중치 사이즈로 변경해준다.
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    # 이미지 정규화
    # astype : 속성
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    #print(frame_reshaped)
    Image_prediction = model.predict(frame_reshaped)
    return Image_prediction

#화면 테두리에 붉은 선을 그려주는 함수
def draw_warning(frame):
    #해당 프레임의 높이값을 가져온다
    height = frame.shape[0]
    #해당 프레임의 넓이값을 가져온다
    width = frame.shape[1]
    #프레임에 프레임 높이와 넓이값과 같은 크기의 붉은 사각형을 그려준다.
    cv2.rectangle(frame ,(0,0),(width,height),(0,0,255),5)

#레이아웃을 적당한 크기로 사이즈를 바꿔준다.
def Layout_resize(img):
    #이미지를 112*64의 크기로 변환한다.
    img = cv2.resize(img, (112,64), interpolation=cv2.INTER_AREA)
    return img



def filter(image , mask):
    rows,cols = image.shape[:2]
    dst = np.zeros((rows,cols),np.float32)
    ycenter , xcenter = mask.shape[0]//2 , mask.shape[1]//2
    for i in range(ycenter, rows-ycenter):
        for j in range(xcenter,cols-xcenter):
            y1,y2 = i-ycenter,i+ycenter+1
            x1, x2 = j - xcenter, j + xcenter + 1
            roi = image[y1:y2, x1:x2].astype("float32")
            tmp = cv2.multiply(roi, mask)
            dst[i, j] = cv2.sumElems(tmp)[0]
    return dst

def filter2(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)                 # 회선 결과 저장 행렬
    xcenter, ycenter = mask.shape[1]//2, mask.shape[0]//2  # 마스크 중심 좌표

    for i in range(ycenter, rows - ycenter):                  # 입력 행렬 반복 순회
        for j in range(xcenter, cols - xcenter):
            sum = 0.0
            for u in range(mask.shape[0]):                    # 마스크 원소 순회
                for v in range(mask.shape[1]):
                    y, x = i + u - ycenter , j + v - xcenter
                    sum += image[y, x] * mask[u, v]           # 회선 수식
            dst[i, j] = sum
    return dst

#프레임에 레이아웃 스테이트에 맞는 레이아웃을 넣어주고 해당 이미지 개선은 진행한다.
def make_layout(frame,Layout_state,original_clicked,original_unclicked,edge_clicked,edge_unclicked,improved_clicked,improved_unclicked):
    if (Layout_state == 0):
        frame[20:84,224:336] =  original_clicked
        frame[20:84,356:468] =  edge_unclicked
        frame[20:84,488:600] =  improved_unclicked
    elif (Layout_state == 1):
        #Canny기법으로 엣지를 찾아낸다
        frame = cv2.Canny(frame,100,255)
        #그레이스케일 이미지를 BGR형태로 반환한다.
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        frame[20:84,224:336] =  original_unclicked
        frame[20:84,356:468] =  edge_clicked
        frame[20:84,488:600] =  improved_unclicked
    elif (Layout_state == 2):
        #BGR값을 CrCb행태로 변환
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        #CrCb의 형태에서 밝기값에 대해서만 접근한다.
        ycrcb_planes = cv2.split(frame_ycrcb)
        #split함수로 인한 반환값이 튜플의 형태로 나와서 리스트로 변경해준다.
        ycrcb_planes = list(ycrcb_planes)
        #히스토그램이퀄라이제이션을 진행한다.
        ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
        #밝기값에 히스토그램이퀄라이제이션한것을 다시 합쳐준다.
        dst_ycrcb = cv2.merge(ycrcb_planes)
        #CrCb형태의 이미지를 다시 BGR형태로 변환해준다.
        frame = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
        frame[20:84,224:336] =  original_unclicked
        frame[20:84,356:468] =  edge_unclicked
        frame[20:84,488:600] =  improved_clicked

def histogram_equalization(frame):
    #BGR값을 CrCb행태로 변환
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        #CrCb의 형태에서 밝기값에 대해서만 접근한다.
        ycrcb_planes = cv2.split(frame_ycrcb)
        #split함수로 인한 반환값이 튜플의 형태로 나와서 리스트로 변경해준다.
        ycrcb_planes = list(ycrcb_planes)
        #히스토그램이퀄라이제이션을 진행한다.
        ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
        #밝기값에 히스토그램이퀄라이제이션한것을 다시 합쳐준다.
        dst_ycrcb = cv2.merge(ycrcb_planes)
        #CrCb형태의 이미지를 다시 BGR형태로 변환해준다.
        frame = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
        return frame
    
def sharpening(frame):
    #sharpening 마스크 가져오기
    mask = [0, -1, 0,
            -1, 5, -1,
            0, -1, 0]
    #마스크를 넘파이의 형태로 변환
    mask = np.array(mask, np.float32).reshape(3, 3)
    #프레임을 BGR의 형태로 나눈다
    frame_planes = cv2.split(frame)
    #튜플형태를 리스트 형태로 변환해준다.
    frame_planes = list(frame_planes)
    #각각의 BGR에대해 마스크 연산을 진행한다.
    frame_planes[0] = filter2(frame_planes[0] , mask)
    frame_planes[1] = filter2(frame_planes[1] , mask)
    frame_planes[2] = filter2(frame_planes[2] , mask)
    #나눠줬던 BGR을 다시 합쳐주는 과정을 거친다.
    dst_frame = cv2.merge(frame_planes)
    return dst_frame