//TODO: add code for read data from lmdb 20160323 19:03

#include <iosfwd>
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::vector;
using std::pair;
using std::make_pair;

using std::cout;
using std::endl;
using std::cerr;

//jin: 20160324 09:14 copied from extract_one_feature.cpp
//template<typename Dtype>  
class WriteDb  
{  
public:  
    void open(string dbName) { db_.open(dbName.c_str()); }  
	void write(const int sz) {db_ << sz << endl;}
 
	void write(const pair<string, float> &p) { db_ << p.first << "  " << p.second << endl;}
	void write(const vector<pair<string, float> > &vp) 
	{
		int sz = vp.size();
		write(sz);
		for(int i=0; i<sz; ++i)
		{
			write(vp[i]);
		}
	}
	
//	void write(const float &data) { db<<data; }    
//	void write(const string &str) { db<<str;  } 
	
	virtual ~WriteDb() {  db_.close(); }  
	
private:	
    std::ofstream db_;  
}; 

//jin: 20160324 13:23 refer to https://github.com/BVLC/caffe/blob/master/include/caffe/util/db.hpp
// and caffe-rc/include/util/data_layers.hpp
enum Mode { READ, WRITE, NEW };
class LMDB
{
public:
	LMDB() : mdb_env_(NULL) { }
	
	virtual ~LMDB() { close(); }
	
	void open(const char *source)
	{
		CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";          
		CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS)  // 1TB
				<< "mdb_env_set_mapsize failed";        
		CHECK_EQ(mdb_env_open(mdb_env_, source, MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";       
		CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS) << "mdb_txn_begin failed";      
		CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS) << "mdb_open failed";        
		LOG(INFO) << "Opening source lmdb1 in " << source << endl;                  
		CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS) << "mdb_cursor_open failed";
	}
	
	void close() {
		if (mdb_env_ != NULL) {
			mdb_dbi_close(mdb_env_, mdb_dbi_);
			mdb_env_close(mdb_env_);
			mdb_env_ = NULL;
		}
	}
	
	void datum(Datum *datum)
	{
		CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_SET_KEY), MDB_SUCCESS) << "mdb_cursor_get failed";
		datum->ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
	}

	void set_key(const string &str){ 
		string keystr = str;
		mdb_key_.mv_size = keystr.size();
		mdb_key_.mv_data = reinterpret_cast<void*>(&keystr[0]); 
	}
	
	string value() {
		CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_SET_KEY), MDB_SUCCESS) << "mdb_cursor_get failed";
		return string(static_cast<const char*>(mdb_value_.mv_data), mdb_value_.mv_size);
	}	
	
//	//copied from https://github.com/BVLC/caffe/blob/master/include/caffe/util/db_lmdb.hpp
//	virtual void SeekToFirst() { Seek(MDB_FIRST); }
//	virtual void Next() { Seek(MDB_NEXT); }
	
//	string key() {
//		return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
//	}
	

	
//	virtual LMDBCursor* NewCursor();
//	virtual LMDBTransaction* NewTransaction();
	
private:
	// LMDB
	MDB_env* mdb_env_;
	MDB_dbi mdb_dbi_;
	MDB_txn* mdb_txn_;
	MDB_cursor* mdb_cursor_;
	MDB_val mdb_key_, mdb_value_;
	
};


//============================== utility functions =================================
int load_strlist(vector<string> &strlist, const char *strlistname)
{ 
    std::ifstream infile(strlistname);
    if(!infile)
    {
        cerr << "Failed to open ifstream " << strlistname << endl;
        return -1;
    }

    string filename;
    int total_num;
    infile >> total_num;

    for(int i=0; i<total_num; ++i)
    {
        infile >> filename;
        strlist.push_back(filename);             
    }

    infile.close();
    return 0;    
}
//end jin: 20160324 09:14


/* Pair (label, confidence) representing a prediction. */
//typedef std::pair<string, float> Prediction;
typedef std::pair<int, float> Prediction;

class Classifier {

public:
    Classifier(const string& model_file,
             const string& trained_file);          

    std::vector<float> predict(const Datum& datum);
   
	//jin: read data from lmdb and its keylist
    std::vector<Prediction> classify(const Datum& datum); 
	//std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);//N = 5

private:
	
	//divide each unsigned char by 256 and assign to input_layer
    void preprocess(const Datum& datum);
	
//    std::vector<float> Predict(const cv::Mat& img);
//    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
//    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
    shared_ptr<Net<float> > net_;
//    cv::Size input_geometry_;
	int width_, height_;
    int num_channels_;
    std::vector<int> labels_;
};

Classifier::Classifier(const string& model_file, const string& trained_file) 
{
//#ifdef CPU_ONLY
//    Caffe::set_mode(Caffe::CPU);
//#else
//    Caffe::set_mode(Caffe::GPU);
//#endif

    /* Load the network. */
    //jin 2016-03-21 18:12:27 
    //net_.reset(new Net<float>(model_file, TEST));
    net_.reset(new Net<float>(model_file));
    Caffe::set_phase(Caffe::TEST);  
  
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)   << "Input layer should have 1 or 3 channels.";
//    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	width_ = input_layer->width();
	height_ = input_layer->height();
	
	labels_.push_back(0);
	labels_.push_back(1);
	
    Blob<float>* output_layer = net_->output_blobs()[0];     
    CHECK_EQ(labels_.size(), output_layer->channels())    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    
    std::sort(pairs.begin(), pairs.end(), PairCompare);


    std::vector<int> result;
    for (int i = 0; i < N; ++i)
    {
        result.push_back(pairs[i].second);
    //   std::cout << pairs[i].second << std::endl;
    }
    return result;
}

/* Return the top 1 predictions. */
std::vector<Prediction> Classifier::classify(const Datum &datum) 
{
    std::vector<float> output = predict(datum);

    //N = std::min<int>(labels_.size(), N);
	int N = 1;
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }

    return predictions;
}

std::vector<float> Classifier::predict(const Datum &datum) 
{
	preprocess(datum);
	
    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];  
    //Blob<float> *output_layer = net_->blob_by_name("prob");

    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}

void Classifier::preprocess(const Datum& datum)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,  height_, width_);
    /* Forward dimension change to all layers. */
    net_->Reshape();
	
	float* input_data = input_layer->mutable_cpu_data();
	const string& data = datum.data();
	
	CHECK_EQ(data.size(), height_ * width_) << "data.size() and input_layer_size not match!";
	for(int i=0; i < data.size(); ++i)
	{
		char tmp = ((i+1)%datum.height() == 0) ? '\n':'\t';
		input_data[i] = (uint8_t)data[i] * 1.0 / 256;
		//printf("%d %6.2f%c", (uint8_t)data[i], input_data[i], tmp);
	}	
}

int main(int argc, char** argv) 
{
	// 0. check args and initialize vars
	const int num_required_args = 6;
	if (argc < num_required_args) {
		LOG(ERROR)<< "Usage: " << argv[0]  << " deploy_prototxt network_caffemodel lmdb_path lmdb_keylist outpath [GPU/CPU] [DEVICE_ID=0]";
		return 1;
	}
	int arg_pos = num_required_args;
	
	if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
		LOG(ERROR)<< "Using GPU";
		uint device_id = 0;
		if (argc > arg_pos + 1) {
			device_id = atoi(argv[arg_pos + 1]);
			CHECK_GE(device_id, 0);
		}
		LOG(ERROR) << "Using Device_id=" << device_id;
		Caffe::SetDevice(device_id);
		Caffe::set_mode(Caffe::GPU);
	} else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}
	Caffe::set_phase(Caffe::TEST);

	arg_pos = 0;  // the name of the executable
    ::google::InitGoogleLogging(argv[0]);
	string model_file(argv[++arg_pos]);  //argv[1]
	
    string trained_file(argv[++arg_pos]); //argv[2]
	
    Classifier classifier(model_file, trained_file);

	const char *src_db_path1 = argv[++arg_pos]; //argv[3];
	
	LMDB lmdb;
	lmdb.open(src_db_path1);
	
	 //used to store images with classification probability in [0, 0.5], (0.5, 0.7], (0.7, 0.9], (0.9, 1]
	vector<std::pair<string, float> > sv1, sv2, sv3, sv4; 
	
    // 1. read keystr from SRC_DB1_LIST into vector db_keystrlist1;
    const char *src_db_keylist1 = argv[++arg_pos]; //argv[4];
	vector<string> db_keylist1;
	
    if(load_strlist(db_keylist1, src_db_keylist1)) { return -1; }
	//jin test
	printf("db_keylist1.size() = %d\n", db_keylist1.size());
	
	// 2. open lmdb1 for reading image data by keys 
	for( int i = 0; i < db_keylist1.size(); ++i )  
    {  
        string keystr1 = db_keylist1[i];
		lmdb.set_key(keystr1);
		//test: print value of key
		
		
		Datum datum;
		lmdb.datum(&datum);
		          
		std::vector<float> output = classifier.predict(datum); 
		
		//jin: only output the probability of each image as a face
		float sc = output[1]; 
		if(sc<=0.5){
			sv1.push_back(make_pair(keystr1, sc));
		}else if(sc<=0.7){
			sv2.push_back(make_pair(keystr1, sc));
		}else if(sc<=0.9){
			sv3.push_back(make_pair(keystr1, sc));
		}else{
			sv4.push_back(make_pair(keystr1, sc));
		}
		
		//test
		//printf("%s  %.4f\n", keystr1.c_str(), sc);
		
		if((i+1)%1000 == 0){
			//LOG(ERROR) << "Processed " << i+1 << " files.";
			printf("%8d", i+1);
		}
	}
        
	// 3. save sv1,...sv4 into files
	printf("\nwrite all classification results into files");
	string outpath(argv[++arg_pos]); //argv[5];
	WriteDb db1, db2, db3, db4;
	
	db1.open(outpath+"/res1.txt");
	db1.write(sv1);
	db2.open(outpath+"/res2.txt");
	db2.write(sv2);
	db3.open(outpath+"/res3.txt");
	db3.write(sv3);
	db4.open(outpath+"/res4.txt");
	db4.write(sv4);
}



//* Wrap the input layer of the network in separate cv::Mat objects
// * (one per channel). This way we save one memcpy operation and we
// * don't need to rely on cudaMemcpy2D. The last preprocessing
// * operation will write the separate channels directly to the input
// * layer. */
//void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
//  Blob<float>* input_layer = net_->input_blobs()[0];

//  int width = input_layer->width();
//  int height = input_layer->height();
//  float* input_data = input_layer->mutable_cpu_data();
//  for (int i = 0; i < input_layer->channels(); ++i) {
//	  //jin 20160323 16:30
//	cv::Mat channel(height, width, CV_32FC1, input_data);
//    //cv::Mat channel(height, width, CV_8UC1, input_data);
//    input_channels->push_back(channel);
//    input_data += width * height;
//  }
//}

//void Classifier::Preprocess(const cv::Mat& img,
//                            std::vector<cv::Mat>* input_channels) 
//{
//    /* Convert the input image to the input image format of the network. */
//    cv::Mat sample;
//    if (img.channels() == 3 && num_channels_ == 1)
//        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//    else if (img.channels() == 4 && num_channels_ == 1)
//        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
//    else if (img.channels() == 4 && num_channels_ == 3)
//        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//    else if (img.channels() == 1 && num_channels_ == 3)
//        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
//    else
//        sample = img;

//    cv::Mat sample_resized;
//    if (sample.size() != input_geometry_)
//        cv::resize(sample, sample_resized, input_geometry_);
//    else
//        sample_resized = sample;

//    cv::Mat sample_float;
//    if (num_channels_ == 3)
//        sample_resized.convertTo(sample_float, CV_32FC3);
//    else
//        sample_resized.convertTo(sample_float, CV_32FC1);

//    //jin 20160323 16:32  divide sample_float by 256  
//    sample_float = sample_float * 0.00390625; //  1/256
//   
//    cv::split(sample_float, *input_channels);

//    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//        == net_->input_blobs()[0]->cpu_data())
//    << "Input channels are not wrapping the input layer of the network.";
//}
