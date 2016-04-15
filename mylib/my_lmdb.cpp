//jin: 20160413 16:57 
//jin: 20160411 12:28 rewrite code for combining two lmdbs, refer to combine_lmdb2.cpp
//jin: 20160405 16:53  try to wrap the lmdb interface with cpp and delete data specified by keys
//	input: lmdb, keystring
//	output: delete the data corresponding to the key
//jin: 20160324 13:23 refer to https://github.com/BVLC/caffe/blob/master/include/caffe/util/db.hpp
// 	and caffe-rc/include/util/data_layers.hpp
//#define PRINT 

#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
using namespace cv; 

#include "my_lmdb.hpp" //jin: move class definitions into hpp 2016/04/12 16:36

//======================================================================class source code
void LMDB::Open(const string &source, Mode mode)
	{
		MDB_CHECK(mdb_env_create(&mdb_env_));
		MDB_CHECK(mdb_env_set_mapsize(mdb_env_, LMDB_MAP_SIZE));
		if (mode == NEW) {
			CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir" << source << "failed";
		}
		int flags = 0;
		if (mode == READ) {
			flags = MDB_RDONLY | MDB_NOTLS;
		}
		int rc = mdb_env_open(mdb_env_, source.c_str(), flags, 0664);
	
	#ifndef ALLOW_LMDB_NOLOCK
		MDB_CHECK(rc);
	#else
		if (rc == EACCES) {
			 cout << "Permission denied. Trying with MDB_NOLOCK ...";
			//LOG(WARNING) << "Permission denied. Trying with MDB_NOLOCK ...";
			// Close and re-open environment handle
			mdb_env_close(mdb_env_);
			MDB_CHECK(mdb_env_create(&mdb_env_));
			// Try again with MDB_NOLOCK
			flags |= MDB_NOLOCK;
			MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
		} else {
			MDB_CHECK(rc);
		}
	#endif
	
		printf("Opened lmdb %s\n", source.c_str()); // LOG(INFO) << "Opened lmdb " << source;
	}

	void Cursor::Imshow(const string &str)
	{
		printf("keystr = %s\n", str.c_str());
		datum_.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);	
	
		const string &data = datum_.data();
	
		int ht = datum_.height(), wd = datum_.width();
		cv::Mat img(ht, wd, CV_8UC1);
	
		for(int i=0; i<ht; i++){
			for(int j=0; j<wd; j++){
				img.at<uint8_t>(i,j) = (uint8_t)data[i*ht+j];
			}
		}
	
		cv::namedWindow( "Image", 1 );//创建窗口
		cv::imshow( "Image", img );//显示图像
	
		cv::waitKey(0); //等待按键
	
		cv::destroyWindow( "Image" );//销毁窗口
		//cv::releaseImage( &img ); //释放图像
	}
//====================================================================== helper
int load_keylist(vector<string> &strlist, const char *strlistname)
{ 
    std::ifstream infile(strlistname);
    if(!infile)
    {
        printf("Failed to open ifstream %s\n", strlistname);
        return -1;
    }

    string keystr;
	float tmp;
    int total_num;
    infile >> total_num;

    //for(int i=0; i<total_num; ++i)
	while(infile >> keystr >> tmp)
    {
        //infile >> keystr ;
        strlist.push_back(keystr);             
    }

    infile.close();
    return 0;    
}

//============================== utility functions ==============================
/*
int bilinear_image_resize(const unsigned char *src, int width, int height,
								unsigned char *dst, int nw, int nh)
{
	double dx, dy, x, y;
	int x1, y1, x2, y2;	
	int i, j;	
	
	unsigned char *pout = dst;
	double f1, f2, f3, f4, f12, f34, val;
	
	
	//输入图像的采样间隔
	dx = (double) (width) / (double) (nw);
	dy = (double) (height) / (double) (nh);
	
	for (j=0; j<nh; ++j) 
	{
		y = j * dy;
		y1 = (int) y;
		y2 = y1 + 1;
		y2 = y2>height-1? height-1 : y2;
		for (i=0; i<nw; ++i) 
		{
			x = i * dx;
			x1 = (int) x;
			x2 = x1 + 1;
			x2 = x2>width-1? width-1 : x2;
			
			f1 = *(src + width * y1 + x1);
			f3 = *(src + width * y1 + x2);
			f2 = *(src + width * y2 + x1);
			f4 = *(src + width * y2 + x2);
			
			f12 = (f1  + (y-y1) * (f2 - f1));
			f34 = (f3  + (y-y1) * (f4 - f3));
			
			//+0.5是为了四舍五入，加0.00001是为了尽量避开取整时的大误差情况
			val = f12 + (x - x1) * (f34 - f12) + 0.5;
			if(val > 255)
				*pout = 255;
			else
				*pout = (unsigned char) val;
			++pout;
		}
	}
	
	return 0;
}
*/
//====================================================================== sub functions

