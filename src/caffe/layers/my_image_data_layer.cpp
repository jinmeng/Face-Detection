//jin 20160322 18:40 copied from http://blog.csdn.net/lingerlanlan/article/details/39400375

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
	MyImageDataLayer<Dtype>::~MyImageDataLayer<Dtype>() {
}

template <typename Dtype>
void MyImageDataLayer<Dtype>::setImgPath(string path, int label)
{
	if(lines_.size() > 0)
		lines_.clear();
	lines_.push_back(std::make_pair(path, label));
}

template <typename Dtype>
void MyImageDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
									vector<Blob<Dtype>*>* top) {
	Layer<Dtype>::SetUp(bottom, top);
	const int new_height  = this->layer_param_.image_data_param().new_height();
	const int new_width  = this->layer_param_.image_data_param().new_width();
	CHECK((new_height == 0 && new_width == 0) ||
		  (new_height > 0 && new_width > 0)) << "Current implementation requires "
				  "new_height and new_width to be set at the same time.";

	/*
	 * 因为下面需要随便拿一张图片来初始化blob。
	 * 因此需要硬盘上有一张图片。
	 * 1 从prototxt读取一张图片的路径，
	 * 2 其实也可以在这里将用于初始化的图片路径写死
	*/

	/*1*/
	//jin
	const string& source = this->layer_param_.image_data_param().source();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	string filename;
	int label;
	while (infile >> filename >> label) {
	  lines_.push_back(std::make_pair(filename, label));
	  break; //jin
	}
	

	/*2*/
	//lines_.push_back(std::make_pair("/home/linger/init.jpg", 1));

	//上面1和2代码可以任意用一段

	lines_id_ = 0;
	// Read a data point, and use it to initialize the top blob. (随便)读取一张图片，来初始化blob
	Datum datum;
	CHECK(ReadImageToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
						   new_height, new_width, &datum));
	// image
	const int crop_size = this->layer_param_.image_data_param().crop_size();
	const int batch_size = 1;//this->layer_param_.image_data_param().batch_size();
	const string& mean_file = this->layer_param_.image_data_param().mean_file();
	if (crop_size > 0) {
		(*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		(*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
						   datum.width());
		prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
							   datum.width());
	}
	LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
			  << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
			  << (*top)[0]->width();
	// label
	(*top)[1]->Reshape(batch_size, 1, 1, 1);
	prefetch_label_.Reshape(batch_size, 1, 1, 1);
	// datum size
	datum_channels_ = datum.channels();
	datum_height_ = datum.height();
	datum_width_ = datum.width();
	datum_size_ = datum.channels() * datum.height() * datum.width();
	CHECK_GT(datum_height_, crop_size);
	CHECK_GT(datum_width_, crop_size);
	// check if we want to have mean
	if (this->layer_param_.image_data_param().has_mean_file()) {
		BlobProto blob_proto;
		LOG(INFO) << "Loading mean file from" << mean_file;
		ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
		data_mean_.FromProto(blob_proto);
		CHECK_EQ(data_mean_.num(), 1);
		CHECK_EQ(data_mean_.channels(), datum_channels_);
		CHECK_EQ(data_mean_.height(), datum_height_);
		CHECK_EQ(data_mean_.width(), datum_width_);
	} else {
		// Simply initialize an all-empty mean.
		data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
	}
	// Now, start the prefetch thread. Before calling prefetch, we make two
	// cpu_data calls so that the prefetch thread does not accidentally make
	// simultaneous cudaMalloc calls when the main thread is running. In some
	// GPUs this seems to cause failures if we do not so.
	prefetch_data_.mutable_cpu_data();
	prefetch_label_.mutable_cpu_data();
	data_mean_.cpu_data();


}

//--------------------------------下面是读取一张图片数据-----------------------------------------------
template <typename Dtype>
void MyImageDataLayer<Dtype>::fetchData() {
	Datum datum;
	CHECK(prefetch_data_.count());
	Dtype* top_data = prefetch_data_.mutable_cpu_data();
	Dtype* top_label = prefetch_label_.mutable_cpu_data();
	ImageDataParameter image_data_param = this->layer_param_.image_data_param();
	const Dtype scale = image_data_param.scale();//image_data_layer相关参数
	const int batch_size = 1;//image_data_param.batch_size(); 这里我们只需要一张图片

	const int crop_size = image_data_param.crop_size();
	const bool mirror = image_data_param.mirror();
	const int new_height = image_data_param.new_height();
	const int new_width = image_data_param.new_width();

	if (mirror && crop_size == 0) {
	    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
				   << "set at the same time.";
	}
	// datum scales
	const int channels = datum_channels_;
	const int height = datum_height_;
	const int width = datum_width_;
	const int size = datum_size_;
	const int lines_size = lines_.size();
	const Dtype* mean = data_mean_.cpu_data();

	for (int item_id = 0; item_id < batch_size; ++item_id) {//读取一图片
	    // get a blob
	    CHECK_GT(lines_size, lines_id_);
	    if (!ReadImageToDatum(lines_[lines_id_].first,
							  lines_[lines_id_].second,
							  new_height, new_width, &datum)) {
			continue;
	    }
	    const string& data = datum.data();
	    if (crop_size) {
			CHECK(data.size()) << "Image cropping only support uint8 data";
			int h_off, w_off;
			// We only do random crop when we do training.
	        h_off = (height - crop_size) / 2;
	        w_off = (width - crop_size) / 2;

	        // Normal copy 正常读取，把裁剪后的图片数据读给top_data
	        for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < crop_size; ++h) {
					for (int w = 0; w < crop_size; ++w) {
						int top_index = ((item_id * channels + c) * crop_size + h)
										* crop_size + w;
						int data_index = (c * height + h + h_off) * width + w + w_off;
						Dtype datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
						top_data[top_index] = (datum_element - mean[data_index]) * scale;
					}
				}
	        }

	    } else {
			// Just copy the whole data 正常读取，把图片数据读给top_data
			if (data.size()) {
				for (int j = 0; j < size; ++j) {
					Dtype datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(data[j]));
					top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
				}
			} else {
				for (int j = 0; j < size; ++j) {
					top_data[item_id * size + j] =
						(datum.float_data(j) - mean[j]) * scale;
				}
			}
	    }
	    top_label[item_id] = datum.label();//读取该图片的标签

	}
}

template <typename Dtype>
// Dtype  //jin
void MyImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {

	//更新input
	fetchData();

	// Copy the data
	caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
			   (*top)[0]->mutable_cpu_data());
	caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
			   (*top)[1]->mutable_cpu_data());

//	return Dtype(0.); //jin
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageDataLayer, Forward);
#endif

INSTANTIATE_CLASS(MyImageDataLayer);

}  // namespace caffe

