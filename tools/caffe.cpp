#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

#ifdef _MSC_VER
#include "caffe/layers/memory_data_layer.hpp"
#include <opencv2/opencv.hpp>
#endif

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(ssd_test_image, "",
    "Optional; an image to use by sddtest");
DEFINE_int32(ssd_label, -1,
    "Optional; label id to output results for, -1 means all");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Parse phase from flags
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
  return caffe::SolverAction::NONE;
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  vector<string> stages = get_stages_from_flags();

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);
  }

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  } else {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;

    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);

#ifdef _MSC_VER
struct ImagePatch {
  int x;
  int y;
  int width;
  int height;
  ImagePatch() :
    x(0), y(0), width(0), height(0) {}
  ImagePatch(int w, int h) :
    x(0), y(0), width(w), height(h) {}
  ImagePatch(int x, int y, int w, int h) :
    x(x), y(y), width(w), height(h) {}
};

struct ObjectDetection {
  bool ignore;
  float score;
  float scale; // image scale at which the face was detected
  int left, top, bottom, right;
  int label;
  ObjectDetection() : ignore(false), score(0.f), label(-1) {}
};

// return 1 if reference should be ignore, 2 if rect should be ignored and 0 if neither should be ignored
int intersect(const ObjectDetection& reference, const ObjectDetection& rect) {

  if (reference.scale < rect.scale) {
    // if reference was found at a smaller scale don't allow any boxes inside of it
    if (rect.left >= reference.left &&
      rect.right <= reference.right &&
      rect.top >= reference.top &&
      rect.bottom <= reference.bottom) {
      return 2;
    }
  }

  const int left = std::max(reference.left, rect.left);
  const int right = std::min(reference.right, rect.right);

  if (right < left)
    return 0;

  const int top = std::max(reference.top, rect.top);
  const int bottom = std::min(reference.bottom, rect.bottom);

  if (bottom < top)
    return 0;

  const int intersectionArea = (right - left + 1) * (bottom - top + 1);
  const int rectArea = (rect.right - rect.left) * (rect.bottom - rect.top);

  const int referenceArea = (reference.right - reference.left) * (reference.bottom - reference.top);
  const int unionArea = referenceArea + rectArea - intersectionArea;

  float threshold_ = 0.2f;

  if ((float)intersectionArea >= (float)unionArea * threshold_) {
    // ignore the least likely face 
    if (rect.scale != reference.scale) {
      return rect.scale < reference.scale ? 1 : 2;
    }
    else {
      return rect.score > reference.score ? 1 : 2; // rectArea > referenceArea ? 1 : 2;
    }
  }
  return 0;
}

void nonMaximaSuppression_(std::map<float, ObjectDetection>& detections)
{
  // Non maxima suppression
  typedef std::map<float, ObjectDetection>::reverse_iterator RFaceItr;
  typedef std::map<float, ObjectDetection>::iterator FaceItr;

  for (RFaceItr i = detections.rbegin(); i != detections.rend(); i++) {

    if (i->second.ignore) continue;

    // find any other overlapping box
    for (RFaceItr j = detections.rbegin(); j != detections.rend(); j++) {
      if (j != i && !j->second.ignore && !i->second.ignore)
      {
        int it = intersect(i->second, j->second);
        if (it == 1) {
          i->second.ignore = true;
          break;
        }
        else if (it == 2) {
          j->second.ignore = true;
        }
      }
    }
  }

  // clean up detections
  for (FaceItr q = detections.begin(); q != detections.end(); ) {
    if (q->second.ignore) {
      detections.erase(q);
      q = detections.begin();
    }
    else {
      q++;
    }
  }
}
#endif

int ssdtest() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

#ifdef _MSC_VER
  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  }
  else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;

    std::vector<int> labels(1, 0);

    cv::Mat oimg = cv::imread(FLAGS_ssd_test_image, CV_LOAD_IMAGE_COLOR);
 
    // ensure image is below max_im_size_
    float detection_threshold = 0.15f;
    int max_im_size = 700;
    bool do_patches = false;
    int patch_overlap = 50;

    boost::shared_ptr<caffe::MemoryDataLayer<float> > memory_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(caffe_net.layer_by_name("data"));

    if (memory_layer == nullptr) {
      memory_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(caffe_net.layer_by_name("memory_data_input")); 
    }

    int net_img_height = memory_layer->height();
    int net_img_width = memory_layer->width();

    LOG(INFO) << "Net input image size " << net_img_width << "x" << net_img_height;

    cv::Mat img;
    float sf = 1.0;
    int pad_x = 0;
    int pad_y = 0;
    float scale_x = 0;
    float scale_y = 0;

    if (!do_patches) {
      scale_x = static_cast<float>(max_im_size) / static_cast<float>(oimg.cols);
      scale_y = static_cast<float>(max_im_size) / static_cast<float>(oimg.rows);
      cv::resize(oimg, img, cv::Size(max_im_size, max_im_size));
    }
    else {
      // resize down to a maximum of detection_img_size_ if needs be
      if (oimg.cols > max_im_size) {
        sf = static_cast<float>(max_im_size) / static_cast<float>(oimg.cols);
      }
      if (oimg.rows > max_im_size) {
        float sf2 = static_cast<float>(max_im_size) / static_cast<float>(oimg.rows);
        if (sf2 < sf) sf = sf2;
      }

      if (sf == 1.0) {
        img = oimg;
      }
      else {
        cv::resize(oimg, img, cv::Size(), sf, sf);
      }

      scale_x = scale_y = sf;

      // make sure the image is at least the size expected by the detector, if not pad
      if (img.cols < net_img_width) {
        pad_x = net_img_width - img.cols;
        cv::Mat padded(img.rows, img.cols + pad_x, img.type());
        padded.setTo(0);
        img.copyTo(padded(cv::Rect(pad_x / 2, 0, img.cols, img.rows)));
        img = padded;
      }
      if (img.rows < net_img_height) {
        pad_y = net_img_height - img.rows;
        cv::Mat padded(img.rows + pad_y, img.cols, img.type());
        padded.setTo(0);
        img.copyTo(padded(cv::Rect(0, pad_y / 2, img.cols, img.rows)));
        img = padded;
      }
      CHECK_GE(img.cols, net_img_width);
      CHECK_GE(img.rows, net_img_height);
    }

    std::map<float, ObjectDetection> detections;
    typedef std::map<float, ObjectDetection>::reverse_iterator RObjItr;

    std::vector<ImagePatch> sub_imgs;

    if (do_patches && (img.cols > net_img_width || img.rows > net_img_height))
    {
      sub_imgs.push_back(ImagePatch(0, 0, img.cols, img.rows));

      int start_patch_idx = (int)sub_imgs.size();

      // build up the sub patches to process
      if (img.cols > net_img_width)
      {
        int startx = 0;
        while (startx < img.cols)
        {
          ImagePatch part;
          part.x = std::max<int>(0, startx - patch_overlap);
          int endx = part.x + net_img_width;
          // allow to distort the box by 10% to avoid processing an extra patch
          if (img.cols - endx < 0.1*net_img_width) {
            endx = img.cols;
          }
          if (img.cols - part.x < net_img_width) {
            part.x = img.cols - net_img_width;
          }
          part.width = endx - part.x;
          part.height = std::min<int>(net_img_height, img.rows);
          startx += part.width;
          sub_imgs.push_back(part);
        }
      }
      else {
        sub_imgs.push_back(ImagePatch(0, 0, net_img_width, std::min<int>(net_img_height, img.rows)));
      }

      if (img.rows > net_img_height)
      {
        int count = (int)sub_imgs.size();

        for (int n = start_patch_idx; n < count; ++n)
        {
          int starty = net_img_height - patch_overlap;
          int last_endy = 0;

          while (last_endy < img.rows)
          {
            ImagePatch part;
            part.y = starty;
            part.x = sub_imgs[n].x;
            part.width = sub_imgs[n].width;
            if (img.rows - part.y < net_img_height) {
              part.y = img.rows - net_img_height;
            }
            int endy = part.y + net_img_height;
            // allow it to squash a box a bit if it avoids having to process another patch
            if (img.rows - endy < 0.1*net_img_height) {
              endy = img.rows;
            }
            part.height = endy - part.y;
            sub_imgs.push_back(part);

            last_endy = part.y + part.height;
            starty = last_endy - patch_overlap;
          }
        }
      }
    }
    else {
      sub_imgs.push_back(ImagePatch(0, 0, img.cols, img.rows));
    }

    Timer netTimer;
    netTimer.Start();

    for (size_t n = 0; n < sub_imgs.size(); n++)
    {
      std::vector<cv::Mat> netimgs;

      cv::Rect rect;
      rect.x = std::max(0, sub_imgs[n].x);
      rect.y = std::max(0, sub_imgs[n].y);
      rect.width = sub_imgs[n].width;
      rect.height = sub_imgs[n].height;

      cv::Mat subimg = img(rect);
      if (subimg.cols != net_img_width || subimg.rows != net_img_height) {
        cv::resize(subimg, subimg, cv::Size(net_img_width, net_img_height));
      }

      netimgs.push_back(subimg);

      LOG(INFO) << "Patch " << n << " x: " << sub_imgs[n].x << " y: " << sub_imgs[n].y << 
                   " w: " << sub_imgs[n].width << " h: " << sub_imgs[n].height;

      memory_layer->AddMatVector(netimgs, labels);

      const vector<Blob<float>*>& result = caffe_net.Forward(&iter_loss);
      loss += iter_loss;
      int idx = 0;

      const float* result_vec = result[0]->cpu_data();

      for (int k = 0; k < result[0]->count(); k += 7) 
      {
        ObjectDetection det;
        det.score = result_vec[k + 2];
        det.label = result_vec[k + 1];

        if (det.score >= detection_threshold && (det.label == FLAGS_ssd_label || FLAGS_ssd_label < 0))
        {
          det.scale = std::min<float>((float)net_img_width / sub_imgs[n].width, 
                                      (float)net_img_height / sub_imgs[n].height);
          det.left = sub_imgs[n].x + result_vec[k + 3] * sub_imgs[n].width;
          det.top = sub_imgs[n].y + result_vec[k + 4] * sub_imgs[n].height;
          det.right = sub_imgs[n].x + result_vec[k + 5] * sub_imgs[n].width;
          det.bottom = sub_imgs[n].y + result_vec[k + 6] * sub_imgs[n].height;

          // resize to original image dimensions
          det.left -= pad_x>>1;
          det.right -= pad_x>>1;
          det.top -= pad_y>>1;
          det.bottom -= pad_y>>1;

          det.left *= 1.0 / scale_x;
          det.right *= 1.0 / scale_x;
          det.top *= 1.0 / scale_y;
          det.bottom *= 1.0 / scale_y;

          detections.insert(std::make_pair(det.score, det));
        }
      }
    } // all imgage parts

    netTimer.Stop();
    LOG(INFO) << "Time to process image " << netTimer.MilliSeconds() << " msec";

    for (RObjItr it = detections.rbegin(); it != detections.rend(); it++) {
      if (it->first < detection_threshold) {
        it->second.ignore = true;
      }
    }

    nonMaximaSuppression_(detections);

    // draw the patches
    /*
    if (do_patches)
    {
      for (size_t n = 0; n < sub_imgs.size(); n++)
      {
        int x = (1.f / sf) * std::max<int>(0, sub_imgs[n].x);
        int y = (1.f / sf) * std::max<int>(0, sub_imgs[n].y);
        int x0 = std::min<int>(oimg.cols, x + (1.f / sf) * sub_imgs[n].width);
        int y0 = std::min<int>(oimg.rows, y + (1.f / sf) * sub_imgs[n].height);
        cv::rectangle(oimg, cv::Point(x,y), cv::Point(x0, y0), cv::Scalar(n*20, 255-(n*20), 0), 4);
      }
    }*/

    // draw the top detections
    int cnt = 0;
    for (RObjItr it = detections.rbegin(); it != detections.rend(); it++, cnt++) {
      if (it->first > detection_threshold) {
        LOG(INFO) << "Box label " << it->second.label << " score " << it->first << 
          " x: " << it->second.left << " y: " << it->second.top;
        cv::Point p0(it->second.left, it->second.top);
        cv::Point p1(it->second.right, it->second.bottom);
        cv::rectangle(oimg, p0, p1, cv::Scalar(0, 0, 255), 4);
      }
      else {
        break;
      }
    }

    imwrite("ssd_detector_result.jpg", oimg);
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
      caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
      caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
        << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
#endif
  return 0;
}
RegisterBrewFunction(ssdtest);

// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
