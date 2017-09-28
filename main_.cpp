#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <dirent.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

HOGDescriptor Hog;

Mat img, imgF, imgsample, imgHSV, imgGreen, imgGreen1, imgGreen2, imgBlue;

int minHogRectCoorX, minHogRectCoorY, maxHogRectCoorX, maxHogRectCoorY;
//int rLowH1=0,rHighH1=10,rLowH2=170,rHighH2=180,rLowS=160,rHighS=255,rLowV=120,rHighV=255; //red
int gLowH1=35,gHighH1=40,gLowH2=41,gHighH2=59,gLowS1=140,gLowS2=69,gHighS=255,gLowV=104,gHighV=255; //green
int bLowH=99,bHighH=121,bLowS=120,bHighS=255,bLowV=57,bHighV=211; //blue
//int yLowH=26,yHighH=32,yLowS=130,yHighS=255,yLowV=150,yHighV=255; //yellow

Size winSize(112,24), blockSize(16,16),blockStride(16,8),cellSize(8,8),computeSize(8,8);

void config(){
    int nbin=6;//orientation bins
    HOGDescriptor hog(winSize,blockSize,blockStride,cellSize,nbin);//define(initialize) the parameters of HOG descriptor
    Hog=hog;
}

void getFiles( string path, vector<string>& files )
{
    DIR  *dir;
    struct dirent  *ptr;
    dir = opendir(path.c_str());
    string pathName;

    while((ptr = readdir(dir)) != NULL){
        if(ptr->d_name[0]!='.'&&ptr->d_name[strlen(ptr->d_name)-4]=='.'){
            files.push_back(pathName.assign(path).append("/").append(string(ptr->d_name)));
        }
    }
}

Mat colorDetector(Mat imgF)
{
    //medianBlur(imgF, imgF, 5); //median filting image with 5*5 size, time=5~6ms

    cvtColor(imgF, imgHSV, COLOR_BGR2HSV); //convert the RGB image to HSV, opencv is BGR time=1s

    //time=6ms
    inRange(imgHSV, Scalar(bLowH,bLowS, bLowV),Scalar(bHighH,bHighS, bHighV),imgBlue); //detecting Blue,
    inRange(imgHSV, Scalar(gLowH1,gLowS1, gLowV),Scalar(gHighH1,gHighS, gHighV),imgGreen1); //detecting green
    inRange(imgHSV, Scalar(gLowH2,gLowS2, gLowV),Scalar(gHighH2,gHighS, gHighV),imgGreen2);

    //inRange(imgHSV, Scalar(rLowH1,rLowS, rLowV),Scalar(rHighH1,rHighS, rHighV),imgRed1); //detecting red
    //inRange(imgHSV, Scalar(rLowH2,rLowS, rLowV),Scalar(rHighH2,rHighS, rHighV),imgRed2);
    //inRange(imgHSV, Scalar(yLowH,yLowS, yLowV),Scalar(yHighH,yHighS, yHighV),imgYellow); // detecting yellow

    //add(imgRed1, imgRed2, imgRed);
    add(imgGreen1, imgGreen2, imgGreen);

    vector<vector<Point> > contoursBlue, contoursGreen; //contoursRed, contoursYellow; //define the 2D point vector to save the coordinate (x,y) of contours

    //time=0ms
    findContours(imgBlue,contoursBlue,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(imgGreen,contoursGreen,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    //findContours(imgRed,contoursRed,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    //findContours(imgYellow,contoursYellow,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    //time=2ms from here to the end
    int jB = 0, jG = 0, jR = 0, jY = 0;
    Rect rectCoor[100], rectCoorBlue[10], rectCoorGreen[10]; //rectCoorRed[10],rectCoorYellow[10];
    for (int i = 0; i < contoursBlue.size(); i++)  //calculate the area, center of block and robot, boundaries/rectangles of block and robot
    {
        double imgBlueAreaBuf = contourArea(contoursBlue[i]); //contours: the points of contours

        if (imgBlueAreaBuf > 5){
            rectCoorBlue[jB] = boundingRect(contoursBlue[i]);
            rectCoor[jB] = rectCoorBlue[jB];
            jB++;
        }
    }
    //----------Green-----------
    for (int i = 0; i < contoursGreen.size(); i++)  //calculate the area, center of block and robot, boundaries/rectangles of block and robot
    {
        double imgGreenAreaBuf = contourArea(contoursGreen[i]); //contours: the points of contours
        if (imgGreenAreaBuf > 5){

            rectCoorGreen[jG] = boundingRect(contoursGreen[i]);
            rectCoor[jB+jG] = rectCoorGreen[jG];
            jG++;
        }
    }
    //---------Red----------
//    for (int i = 0; i < contoursRed.size(); i++)  //calculate the area, center of block and robot, boundaries/rectangles of block and robot
//    {
//        double imgRedAreaBuf = contourArea(contoursRed[i]); //contours: the points of contours
//
//        if (imgRedAreaBuf > 80){
//            rectCoorRed[jR] = boundingRect(contoursRed[i]);
//            rectCoor[jB+jG+jR] = rectCoorRed[jR];
//            jR++;
//        }
//    }
    //----------Yellow----------
//    for (int i = 0; i < contoursYellow.size(); i++)  //calculate the area, center of block and robot, boundaries/rectangles of block and robot
//    {
//        double imgYellowAreaBuf = contourArea(contoursYellow[i]); //contours: the points of contours
//
//        if (imgYellowAreaBuf > 100){
//            rectCoorYellow[jY] = boundingRect(contoursYellow[i]);
//            rectCoor[jB+jG+jR+jY] = rectCoorYellow[jY];
//            jY++;
//        }
//    }
    //---------define include distance-----------
    int num = jB+jG;
    //int num = jB+jG+jR+jY;
    int distanceBox[num][num], distanceBoxSum[num], numBox[num], minDistanceBox[num], min2DistanceBox[num],rectBoxHeight = 0, rectBoxHeightMax = 0;
    for (int i = 0; i < num; i++) //calculating the suitable(medium) value of height
    {
        if (rectCoor[i].height > rectBoxHeightMax)
        {
            rectBoxHeight = rectBoxHeightMax; // set this value as the height of box
            rectBoxHeightMax = rectCoor[i].height;
        }
        else if (rectCoor[i].height > rectBoxHeight)
            rectBoxHeight = rectCoor[i].height;
    }

    for (int j = 0; j < num; j++) //calculating the value of minimum and the second minimum distance for each box
    {
        minDistanceBox[j] = 800;
        min2DistanceBox[j] = 800;
        for (int x = 0; x < num; x++)
        {
            if (j != x)
            {
                distanceBox[j][x] = min(abs(rectCoor[j].tl().x - rectCoor[x].br().x),abs(rectCoor[j].br().x - rectCoor[x].tl().x));

                if (distanceBox[j][x] < minDistanceBox[j])
                {
                    min2DistanceBox[j] = minDistanceBox[j]; //the second minimum distance
                    minDistanceBox[j] = distanceBox[j][x]; //the minimun distance
                }
                else if (distanceBox[j][x] < min2DistanceBox[j])
                {
                    min2DistanceBox[j] = distanceBox[j][x];
                }
            }
        }
        distanceBoxSum[j] = minDistanceBox[j] + min2DistanceBox[j];
    }

    for (int i =0; i < num; i++) //sequence from minimum distance to maximum distance
    {
        numBox[i] = 0;
        for (int j=0; j < num; j++)
        {
            if (i != j) // get the Box[i] sequence
            {
                if (distanceBoxSum[i] > distanceBoxSum[j])
                    numBox[i]+=1; //numBox[i] = numBox[i] +1, save the number
                if (distanceBoxSum[i] == distanceBoxSum[j])
                {
                    if (minDistanceBox[i] >= minDistanceBox[j]) //always have the same distance between two points each other
                        numBox[i]+=1; //
                }
            }
        }
    }
    //-------------difine the robot------------
    int lastnum = num, robNum, minRectCoorX[num], minRectCoorY[num], maxRectCoorX[num], maxRectCoorY[num];
    for (robNum = 0; lastnum >= 2 && robNum < num; robNum++)
    {
        int minNumBox=100;
        for (int k = 0; k <num; k++) //get the minNumBox between the rest
        {
            minNumBox = min(numBox[k], minNumBox);
        }
        for (int i = 0; i < num; i++) //get the coordination of rectangle of robot from boxes
        {
            if (numBox[i] == minNumBox) //find the minimum one between the rest (usually it is 0 when 1 robot)
            {
                lastnum --;
                if (num > 2) //when robot only have 2 boxes at least, just combine the two boxes
                    numBox[i] = 100; //make it not included in the rest
                minRectCoorX[robNum] = rectCoor[i].tl().x;
                minRectCoorY[robNum] = rectCoor[i].tl().y;
                maxRectCoorX[robNum] = rectCoor[i].br().x;
                maxRectCoorY[robNum] = rectCoor[i].br().y;
                int bufnum = 0, jBox[50] = {0};
                for (int j = 0; j < num; j++) //calculating the coordination of rectangle incluing boxes belong to the distance area
                {
                    //-------------the first threshold condition-------------------
                    if (j != i && numBox[j] != 100 && distanceBox[i][j] < 5.5 * rectBoxHeight) //3.4, 3.5, 4.5, (4.3) justify if the box belong to the same robot by distance of boxeswith the center box
                    {
                        jBox[bufnum] = j;
                        lastnum --;
                        bufnum ++; //the number of boxes that match the threshold of (distanceBox[i][j] < 3.4 * rectBoxHeight)
                    }
                    //----calculating the max distance between boxes after the first threshold condition, preparing for next--------
                    if (j == num - 1 && bufnum >= 1) //bufnum >= 1 (it have two candidate at least)
                    {
                        int maxBoxDisOut[num], max_in_out[num][num],maxBoxDisOutNum[num];
                        for (int buf = 0; buf < bufnum; buf++) //calculating the max distance between boxes in jBox[bufnum]
                        {
                            maxBoxDisOut[jBox[buf]] = 0;
                            int rectCoor_tl_br, rectCoor_br_tl;
                            if (bufnum == 1) // one other box and one center box
                            {
                                rectCoor_tl_br = abs(rectCoor[i].tl().x - rectCoor[jBox[0]].br().x); //calculating the inside or outside distance between the same boxes
                                rectCoor_br_tl = abs(rectCoor[i].br().x - rectCoor[jBox[0]].tl().x); //calculating the inside or outside distance between the same boxes
                                maxBoxDisOut[jBox[0]] = min(rectCoor_tl_br,rectCoor_br_tl); //max, min
                            }
                            else
                            {
                                for (int buff = 0; buff < bufnum; buff++)
                                {
                                    rectCoor_tl_br = abs(rectCoor[jBox[buf]].tl().x - rectCoor[jBox[buff]].br().x); //calculating the inside or outside distance between the same boxes
                                    rectCoor_br_tl = abs(rectCoor[jBox[buf]].br().x - rectCoor[jBox[buff]].tl().x); //calculating the inside or outside distance between the same boxes
                                    max_in_out[jBox[buf]][jBox[buff]] = min(rectCoor_tl_br,rectCoor_br_tl); //max,min
                                    if (max_in_out[jBox[buf]][jBox[buff]] > maxBoxDisOut[jBox[buf]])
                                    {
                                        maxBoxDisOut[jBox[buf]] = max_in_out[jBox[buf]][jBox[buff]];
                                        maxBoxDisOutNum[buf] = jBox[buff];
                                    }
                                }
                            }
                        }
                        //bufnum >1 guarantte the robot have center box and two other box (bufnum=2) at least, or not go to compare center box and another one box
                        if (bufnum >= 2)
                        {
                            int delNum = 0;
                            for (int bufff = 0; bufff < bufnum; bufff++) //compare the max distance (robot size from left to right) of boxes in jBox[bufnum]
                            {
                                if (maxBoxDisOut[jBox[bufff]] < 6.6 * rectBoxHeight) //(6.2)if > the length of robot, delete far one, get the near one as rectangle
                                {
                                    minRectCoorX[robNum] = min(rectCoor[jBox[bufff]].tl().x, minRectCoorX[robNum]);
                                    minRectCoorY[robNum] = min(rectCoor[jBox[bufff]].tl().y, minRectCoorY[robNum]);
                                    maxRectCoorX[robNum] = max(rectCoor[jBox[bufff]].br().x, maxRectCoorX[robNum]);
                                    maxRectCoorY[robNum] = max(rectCoor[jBox[bufff]].br().y, maxRectCoorY[robNum]);
                                    numBox[jBox[bufff]] = 100; //set a constant not zero and more than all of the numBox
                                }
                                else if (distanceBox[i][jBox[bufff]] < distanceBox[i][maxBoxDisOutNum[bufff]]) //always have two boxes match this condition at the same time, choice one of them
                                {
                                    minRectCoorX[robNum] = min(rectCoor[jBox[bufff]].tl().x, minRectCoorX[robNum]);
                                    minRectCoorY[robNum] = min(rectCoor[jBox[bufff]].tl().y, minRectCoorY[robNum]);
                                    maxRectCoorX[robNum] = max(rectCoor[jBox[bufff]].br().x, maxRectCoorX[robNum]);
                                    maxRectCoorY[robNum] = max(rectCoor[jBox[bufff]].br().y, maxRectCoorY[robNum]);
                                    numBox[jBox[bufff]] = 100; //set a constant not zero and more than all of the numBox
                                }
                                else
                                {

                                    minRectCoorX[robNum] = min(rectCoor[maxBoxDisOutNum[bufff]].tl().x, minRectCoorX[robNum]);
                                    minRectCoorY[robNum] = min(rectCoor[maxBoxDisOutNum[bufff]].tl().y, minRectCoorY[robNum]);
                                    maxRectCoorX[robNum] = max(rectCoor[maxBoxDisOutNum[bufff]].br().x, maxRectCoorX[robNum]);
                                    maxRectCoorY[robNum] = max(rectCoor[maxBoxDisOutNum[bufff]].br().y, maxRectCoorY[robNum]);
                                    numBox[maxBoxDisOutNum[bufff]] = 100;
                                    delNum ++;
                                }
                            }
                            lastnum = lastnum + delNum; //plus for the cancelled more one
                            bufnum = bufnum - delNum;
                        }
                        else //compare center box and another one box, when bufnum = 1
                        {
                            if (maxBoxDisOut[jBox[0]] < 6.6 * rectBoxHeight) //the length of robot 9.4 ->6.2
                            {
                                minRectCoorX[robNum] = min(rectCoor[jBox[0]].tl().x, minRectCoorX[robNum]);
                                minRectCoorY[robNum] = min(rectCoor[jBox[0]].tl().y, minRectCoorY[robNum]);
                                maxRectCoorX[robNum] = max(rectCoor[jBox[0]].br().x, maxRectCoorX[robNum]);
                                maxRectCoorY[robNum] = max(rectCoor[jBox[0]].br().y, maxRectCoorY[robNum]);
                                numBox[jBox[0]] = 100; //set a constant not zero and more than all of the numBox
                            }
                            else //just one center to rest
                            {
                                robNum --;
                            }
                        }
                    }
                }
            }
        }
    }
    //Mat imgsample;
    for (int i = 0; i < robNum; i++)
    {
        //int imgRows = img.rows, imgCols = img.cols;

        rectangle(img, Point(4*minRectCoorX[i],4*minRectCoorY[i]),Point(4*maxRectCoorX[i],4*maxRectCoorY[i]),Scalar(0,0,255),1);

        int imgSamX = 4*minRectCoorX[i]-5;
        int imgSamY = 4*minRectCoorY[i]-5;
        int imgSamHeight = 4 * (maxRectCoorX[i] - minRectCoorX[i]) + 10;
        int imgSamWieth = 4 * (maxRectCoorY[i] - minRectCoorY[i]) + 10;

        minHogRectCoorX = 4*minRectCoorX[i]-5;
        minHogRectCoorY = 4*minRectCoorY[i]-5;
        maxHogRectCoorX = 4*maxRectCoorX[i] + 5;
        maxHogRectCoorY = 4*maxRectCoorY[i] + 5;

        Rect imgSamRect(imgSamX, imgSamY, imgSamHeight, imgSamWieth);
        imgsample = img(imgSamRect);
        resize (imgsample, imgsample, Size(112,24), 0, 0, CV_INTER_AREA);
        //imwrite("/Users/lan/Desktop/Papers/FirstConf/experiments/crop_samples/pos_samples/0920autohard30_0503_800_600/auto0918pm.bmp",imgsample);
        //cout << "----- " << imgsample.cols << "-----" << imgsample.rows << endl;
        //imshow("test", imgsample);
        //imgsample.copyTo(imgsample);
        //imgsample = img(Range(4*minRectCoorX[i],4*minRectCoorY[i]), Range(4*maxRectCoorX[i],4*maxRectCoorY[i]));
        //int robCenterCoorX = 2*(minRectCoorX[i] + maxRectCoorX[i]);
        //int robCenterCoorY = 2*(minRectCoorY[i] + maxRectCoorY[i]);
        //circle(img,Point(robCenterCoorX,robCenterCoorY),3,Scalar(0,255,0),4);
        //char textRobCenterCoor[64], textDistance[64];
        //snprintf(textRobCenterCoor, sizeof(textRobCenterCoor),"(%d,%d)",robCenterCoorX,robCenterCoorY);
        //putText(img, textRobCenterCoor, Point(robCenterCoorX + 10,robCenterCoorY+3),FONT_HERSHEY_DUPLEX,0.4,Scalar(0,255,0),1);

//        int leftLine = 0.4 * img.cols;
//        int rightLine = 0.6 * img.cols;
//        if (robCenterCoorX < leftLine)
//        {
//            int distance = leftLine - robCenterCoorX;
//            snprintf(textDistance, sizeof(textDistance),"L:%d",distance);
//            putText(img, textDistance, Point(0.2*img.cols,15),FONT_HERSHEY_DUPLEX,0.5,Scalar(0,255,0),1);
//            cout << "Left : " << distance << endl;
//        }
//
//        if (robCenterCoorX > rightLine)
//        {
//            int distance = robCenterCoorX - rightLine;
//            snprintf(textDistance, sizeof(textDistance),"R:%d",distance);
//            putText(img, textDistance, Point(0.8*img.cols,15),FONT_HERSHEY_DUPLEX,0.5,Scalar(0,255,0),1);
//            cout << "Right : " << distance << endl;
//        }

        //line(img, Point(4*minRectCoorX[i],4*minRectCoorY[i]), Point(4*maxRectCoorX[i],4*maxRectCoorY[i]),Scalar(0,255,0),1);
        //line(img, Point(4*minRectCoorX[i],4*maxRectCoorY[i]), Point(4*maxRectCoorX[i],4*minRectCoorY[i]),Scalar(0,255,0),1);
        //line(img, Point(leftLine,0), Point(leftLine,img.rows), Scalar(0,255,0),1);
        //line(img, Point(rightLine,0), Point(rightLine,img.rows), Scalar(0,255,0),1);

    }
    //imshow("image", img);
    //waitKey(0);
    return imgsample;
}

int main() {

    vector<string> file_names;
    struct timeval timeStart, timeEnd;
    double timeDiff;
    getFiles("/Users/lan/Desktop/TarReg/svm/crop_samples/tobecroped/49_0502_800_600/",file_names); //49_0502_800_600
    std::cout << "reading the testing image...." << std::endl;

    config();
    int DescriptorDim = Hog.getDescriptorSize();

    CvSVM svm;//SVM classifier
    cout<<"------loading SVM classifier------"<<endl;
    svm.load("/Users/lan/Desktop/Papers/FirstConf/experiments/training/color_D6.xml");//从XML文件读取训练好的SVM模型
    Mat single_sampleFeatureMat= Mat::zeros(1,DescriptorDim, CV_32FC1);;//单个样本的特征向量组成的矩阵，行数等于1，列数等于HOG描述子维数

    for(int fi=0;fi<file_names.size();fi++)
    {
        img = imread(file_names[fi]);
        int imgFcols = img.cols/4;
        int imgFrows = img.rows/4;
        gettimeofday(&timeStart,NULL);
        resize(img, imgF, Size(imgFcols,imgFrows)); //Size(cols, rows)
        //gettimeofday(&timeStart,NULL);
        Mat objLike = colorDetector(imgF);
        //gettimeofday(&timeEnd,NULL);
        //timeDiff = 1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec; //tv_sec: value of second, tv_usec: value of microsecond
        //timeDiff/=1000;
        //cout << "Time for colorDetector : " << timeDiff << " ms -----------------" << endl;
        //cout << "objLike.cols = " << objLike.cols << "objLike.rows = " << objLike.rows << endl;
        //gettimeofday(&timeStart,NULL);
        vector<float> descriptors; //define hog descriptor vector
        //gettimeofday(&timeStart,NULL);
        Hog.compute(objLike,descriptors,computeSize); //compute the hog descriptors计算HOG描述子,src=images,computsSize=winStride
        //gettimeofday(&timeEnd,NULL);
        //timeDiff = 1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec; //tv_sec: value of second, tv_usec: value of microsecond
        //timeDiff/=1000;
        //cout << "Time for Hog.compute: " << timeDiff << " ms -----------------" << endl;
        for(int k=0; k<DescriptorDim; k++)
        {
            single_sampleFeatureMat.at<float>(0,k) = descriptors[k]; //the kth element in the descriptor, 样本的特征向量中的第k个元素
        }

        //gettimeofday(&timeStart,NULL);
        int predict_result=svm.predict(single_sampleFeatureMat); //judgement for current sample 判断当前样本
        gettimeofday(&timeEnd,NULL);
        timeDiff = 1000000*(timeEnd.tv_sec-timeStart.tv_sec)+timeEnd.tv_usec-timeStart.tv_usec; //tv_sec: value of second, tv_usec: value of microsecond
        timeDiff/=1000;
        //cout << "Time for svm.predict: " << timeDiff << " ms -----------------" << endl;
        cout << "Computing Time : " << timeDiff << " ms -----------------" << endl;
        if (predict_result == 1) //识别目标
        {
            rectangle(img, Point(minHogRectCoorX,minHogRectCoorY),Point(maxHogRectCoorX,maxHogRectCoorY),Scalar(0,255,0),2);
            //cout << "----------bingo------------" << endl;
        }
        if (predict_result == -1)
        {
            cout << "----------wrong------------" << endl;
        }

        imshow("image", img);

        while(true){
            int k=cvWaitKey(10);

            if(k=='d'){
                break;
            }
            if(k=='a'){
                fi=(fi-2)<0?-1:(fi-2);
                break;
            }
        }
    }

    return 0;
}