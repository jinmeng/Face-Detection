#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <stdarg.h>
#include <sys/stat.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#define DEBUG_WUHAO 0
//#include "caffe/debugtool.hpp"

static int _MAX_PATH = 1024;

#define ACCESS access
#define MKDIR(a) mkdir((a), 0755)

using namespace caffe;  // NOLINT(build/namespaces)

static int create_directory(const char *directory)
{
  int i;
  int len;
  char dir[_MAX_PATH], temp_dir[_MAX_PATH];

  memcpy(temp_dir, directory, _MAX_PATH);

  len = (int)strlen(temp_dir);
  for(i = 0; i < len; i++) {
    if(temp_dir[i] == '\\')
      temp_dir[i] = '/';
  }
  if(temp_dir[len - 1] != '/') {
    temp_dir[len] = '/';
    temp_dir[len + 1] = 0;
    len++;
  }
  memset(dir, 0, _MAX_PATH);
  for(i = 0; i < len; i++) {
    dir[i] = temp_dir[i];
    if(temp_dir[i] == '/') {
      if(i > 0) {
        if(temp_dir[i - 1] == ':')
          continue;
        else {
          if(ACCESS(dir, 0) == 0)
            continue;
          else {
            /* create it */
            if(MKDIR(dir) != 0)
              return -1;
          }
        }
      }
    }
  }

  return 0;
}

static int get_dir_from_filename(const char *file_name, char *dir)
{
  int len;
  int i;

  len = (int) strlen(file_name);
  for(i = len - 1; i >= 0; i--) {
    if(file_name[i] == '\\' || file_name[i] == '/') {
      break;
    }
  }
  strcpy(dir, file_name);
  dir[i + 1] = 0;
  return 0;
}

static int create_file(const char *file_name, const char type)
{
  FILE *fp;
  char dir[_MAX_PATH];
  char mode[5];

  if(type == 'b') {
    strcpy(mode, "wb");
  }
  else {
    strcpy(mode, "w");
  }

  fp = fopen(file_name, mode);
  if(fp == NULL) {
    get_dir_from_filename(file_name, dir);
    create_directory(dir);
    fp = fopen(file_name, mode);
    if(fp == NULL)
      return -1;
  }
  fclose(fp);

  return 0;
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {

  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {

  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetLogDestination(0, "./log/");

  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(INFO)<<
    "Usage: detector_scan net_prototxt net_model img_dir img_list save_dir gpu_id";
    return 1;
  }

  const float SCALE_STEP = 0.5;    //const float SCALE_STEP = 0.95;
  int ret = 0;

  // set gpu id
  uint device_id = 0;
  device_id = atoi(argv[6]);
  CHECK_GE(device_id, 0);
  LOG(INFO) << "Using Device_id=" << device_id;
  Caffe::SetDevice(device_id);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TEST);

  // load list file
  std::vector<std::string> img_paths;
  std::string path;
  std::ifstream ifs(argv[4]);
  if (!ifs) {
    LOG(FATAL) << "Error open file: " << string(argv[4]);
  } else {
    LOG(INFO) << "File " << string(argv[4]) << " opened.";
  }
  while(ifs >> path) {
    img_paths.push_back(path);
  }
  LOG(INFO) << "Loaded image list: " << img_paths.size() << " images.";
  ifs.close();

  // get img_dir, append '/' if necessary
  std::string img_dir = argv[3];
  if (img_dir[img_dir.length() - 1] != '/') {
    img_dir += "/";
  }

  // get save_dir, mkdir
  std::string save_dir = argv[5];
  ret = create_directory(argv[5]);
  if (ret != 0) {
    LOG(FATAL) << "create dir " << argv[5] << " failed.";
  }

  // init Net
  NetParameter net_param;
  ReadProtoFromTextFile(argv[1], &net_param);
  Net<float> caffe_net(net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_net.CopyTrainedLayersFrom(trained_net_param);
  vector<Blob<float>*> dummy_blob_input_vec;

  std::string data_blob_name = net_param.input(0);
  LOG(INFO) << "data blob name: " << data_blob_name;

  const boost::shared_ptr<Blob<Dtype> > data_blob =
              caffe_net.blob_by_name(data_blob_name);
  const boost::shared_ptr<Blob<Dtype> > argmax_blob =
              caffe_net.blob_by_name("argmax");
  if (!argmax_blob) {
    LOG(FATAL) << "no argmax blob found.";
  }

  LOG(INFO) << data_blob->num() << " "
            << data_blob->channels() << " "
            << data_blob->width() << " "
            << data_blob->height() << " ";

  const int N_DIM = 4;
  int input_dim[N_DIM];
  int i = 0;

  for (i = 0; i < N_DIM; i++) {
    input_dim[i] = net_param.input_dim(i);
  }

  LOG(INFO) << "net input dim: "
            << input_dim[0] << ", "
            << input_dim[1] << ", "
            << input_dim[2] << ", "
            << input_dim[3];

  //
  std::string result_fn = save_dir + "/results.txt";
  FILE* fp_res = fopen(result_fn.c_str(), "w");
  if (fp_res == 0) {
    LOG(FATAL) << "Error opening result file: " << result_fn;
  }
  fprintf(fp_res, "img_path scale new_height new_width y x prob\n");

  std::string img_path;
  float scale = 0.0;
  int batch_size = data_blob->num();
  int net_width = data_blob->width();
  int net_height = data_blob->height();
  int data_dim = data_blob->count() / data_blob->num();
  vector<int> pos;
  pos.resize(batch_size * 2);

  LOG(INFO) << "data_dim: " << data_dim;

  for (i = 0; i < img_paths.size(); ++i) {
    path = img_paths[i];
    img_path = img_dir + path;

    LOG(INFO) << "image path: " << path;
    std::string save_fn = save_dir + "/" + path;
    // LOG(INFO) << save_fn;
    char dir[_MAX_PATH];
    get_dir_from_filename(save_fn.c_str(), dir);
    create_directory(dir);

    int cv_read_flag = CV_LOAD_IMAGE_GRAYSCALE; // CV_LOAD_IMAGE_COLOR
    cv::Mat cv_img = cv::imread(img_path, cv_read_flag);
    if (!cv_img.data) {
      LOG(FATAL) << "Could not open or find file " << img_path;
      return false;
    }

    int net_input_size = std::min(input_dim[2], input_dim[3]);
    int max_face_size = std::min(cv_img.rows, cv_img.cols);
    int min_face_size = net_input_size;
    float min_scale = float(net_input_size) / float(max_face_size);
    float max_scale = float(net_input_size) / float(min_face_size);

    LOG(INFO) << "original size: " << cv_img.rows << ", " << cv_img.cols
              << " scale: " << min_scale << " -- " <<  max_scale;
    //min_scale--->max_scale  jin
    for (scale = max_scale; scale >= min_scale; scale *= SCALE_STEP) {
      cv::Mat cv_img_scaled;
      int new_width = round(cv_img.cols * scale);
      int new_height = round(cv_img.rows * scale);
      cv::resize(cv_img, cv_img_scaled, cv::Size(new_width, new_height));

      LOG(INFO) << "scale: " << scale << " new size(w, h): " << new_width << ", " << new_height;

      Dtype* top_data = data_blob->mutable_cpu_data();
      const Dtype* argmax_data = argmax_blob->cpu_data();
      int cnt = 0;

      for (int yoff = 0; yoff < cv_img_scaled.rows - net_height + 1; yoff+=8) {
        for (int xoff = 0; xoff < cv_img_scaled.cols - net_width + 1; xoff+=8) {

          for (int y = 0; y < net_height; ++y) {
            for (int x = 0; x < net_width; ++x) {
              top_data[cnt * data_dim + y * net_width + x] = Dtype(cv_img_scaled.at<uchar>(yoff + y, xoff+ x)) / Dtype(255);
            }
          }
          pos[cnt * 2] = yoff;
          pos[cnt * 2 + 1] = xoff;
          cnt++;
          bool islast = ((yoff == cv_img_scaled.rows - net_height) &&
                         (xoff == cv_img_scaled.cols - net_width));

          if ((cnt == batch_size) || (islast)) {

#if 0// DEBUG_WUHAO
            if (new_height == 25) {
              DebugTool<Dtype> dbg;
              dbg.open("data.bin");
              dbg.write_blob("data", *data_blob, 0);
              dbg.close();
            }
#endif
            caffe_net.Forward(dummy_blob_input_vec);
            top_data = data_blob->mutable_cpu_data();
            argmax_data = argmax_blob->cpu_data();
            int argmax_dim = argmax_blob->count()/argmax_blob->num();
            
            for (int bi = 0; bi < cnt; ++bi) {
              for (int di = 0; di < argmax_dim / 2; ++ di) {

                int pred = argmax_data[bi * argmax_dim + di * 2];
                if (pred == 1) {
                  Dtype prob = argmax_data[bi * argmax_dim + di * 2 + 1];
                  fprintf(fp_res, "%s %f %d %d %d %d %f\n", path.c_str(), scale, new_height, new_width,
                          pos[bi * 2], pos[bi * 2 + 1], prob);
                  //printf("rect: %d, %d, %d, %d\n", pos[bi * 2 + 1],  pos[bi * 2], net_width, net_height);
                  cv::Mat img_cropped = cv_img_scaled(cv::Rect(pos[bi * 2 + 1], pos[bi * 2], net_width, net_height));
                  char fn_buf[1024];
                  snprintf(fn_buf, 1024, "_%.6f_%03d_%03d.bmp", scale, pos[bi * 2], pos[bi * 2 + 1]);
                  std::string save_img_fn = save_dir + "/" + path.substr(0, path.length()-4) + fn_buf;
                  //std::cout << save_img_fn << std::endl;
                  LOG(INFO) << save_img_fn << " " << prob;
                  cv::imwrite(save_img_fn.c_str(), img_cropped);
                }
                //std::cout << argmax_data[bi * argmax_dim + di] << " ";
              }
              //std::cout << std::endl;
            }
#if 0 // DEBUG_WUHAO
            if (new_height == 25) {
              std::string input_str;
              std::cout << "pause...";
              std::cin >> input_str;
            }
#endif
            cnt = 0;
          }
        }
      }
    }
  }

  fclose(fp_res);
  return 0;
}

