#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/face.hpp"
#include <boost/filesystem.hpp>

using namespace cv;
using namespace cv::face;
using namespace std;
using namespace boost::filesystem;

bool face_recognition(string image_db, string image_test)
{
	// get the frontal face 
	string cascade = "../src/haarcascade_frontalface_default.xml";
	CascadeClassifier face_cascade;

	// check classifer if loaded
	if(!face_cascade.load(cascade))
	{
		cout<<"Error in loading face cascade classifier"<<endl;
		return -1;
	}

	vector<Mat> images;
	vector<int> labels;

	// for the test images with person 1
	string test_path = "../src/test";

	boost::filesystem::path p(test_path);

	directory_iterator it{p};
	directory_iterator end_it;

	while(it!=end_it)
	{
		string dir = it->path().string(); 
		Mat img = imread(dir,IMREAD_COLOR);
		Mat gray;
		cvtColor(img,gray, COLOR_RGB2GRAY);
		*it++;
		equalizeHist(gray, gray);
		Mat face;
		vector<Rect> faces;
	
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

		for(size_t i = 0; i < faces.size(); i++)
    		{
        		rectangle(img, faces[i], Scalar(0, 0, 0), 1);
        				
			face = img(faces[i]);
	    		Size size(150, 150);
			resize(face,face,size);	
			cvtColor(face,gray, COLOR_RGB2GRAY);			
			
			images.push_back(gray);
			labels.push_back(0);				
    		}

	}

	// for the test images with person 2
	boost::filesystem::path pp(image_db);

	directory_iterator itr{pp};
	directory_iterator end_itr;

	while(itr!=end_itr)
	{
		string dir = it->path().string();
		Mat img = imread(dir,IMREAD_COLOR);
		
		Mat gray;
		cvtColor(img, gray, COLOR_RGB2GRAY);
		*itr++;
		equalizeHist(gray, gray);
		Mat face;
		vector<Rect> faces;
	
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

		for(size_t i = 0; i < faces.size(); i++)
    		{
        		rectangle(img, faces[i], Scalar(0, 0, 0), 1);
        				
			face = img(faces[i]);
	    		Size size(150, 150);
			resize(face,face,size);	
			cvtColor(face, gray, COLOR_RGB2GRAY);			

			images.push_back(gray);
			labels.push_back(1);				
    		}
	}


	// for image test 
	Mat test_image = imread(image_test, IMREAD_COLOR);
	Mat gray;
	cvtColor(test_image, gray, COLOR_RGB2GRAY);	

	vector<Rect> faces;
	Mat face;

	face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

	for(size_t i = 0; i < faces.size(); i++)
    	{
        	rectangle(test_image, faces[i], Scalar(0, 0, 0), 1);
        			
		face = test_image(faces[i]);
	    	Size size(150, 150);
		resize(face,face,size);	
		cvtColor(face, gray, COLOR_RGB2GRAY);				
    	}


	// FisherFace Matching 

	Ptr<FisherFaceRecognizer> fisher = FisherFaceRecognizer::create();
    	fisher->train(images, labels);

	int fish = fisher->predict(gray);

	// EigenFace Matching
	
	Ptr<EigenFaceRecognizer> eigener = EigenFaceRecognizer::create();
    	eigener->train(images, labels);
	
	int eigen = eigener->predict(gray);

	// LBPH Matching

	Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    	model->train(images, labels);
    
	int lbph = model->predict(gray);

	return (fish || eigen || lbph) ? true : false;
}
 
