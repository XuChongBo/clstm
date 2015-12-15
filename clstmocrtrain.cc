#include "pstring.h"
#include "clstm.h"
#include "clstmhl.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include <iostream>
#include <set>
#include <regex>

#include "multidim.h"
#include "pymulti.h"
#include "extras.h"

using namespace Eigen;
using namespace ocropus;
using namespace pymulti;
using std::vector;
using std::map;
using std::make_pair;
using std::shared_ptr;
using std::unique_ptr;
using std::cout;
using std::ifstream;
using std::set;
using std::to_string;
using std_string = std::string;
using std_wstring = std::wstring;
using std::regex;
using std::regex_replace;
#define string std_string
#define wstring std_wstring

string basename(string s) {
  int start = 0;
  for (;;) {
    auto pos = s.find("/", start);
    if (pos == string::npos) break;
    start = pos + 1;
  }
  auto pos = s.find(".", start);
  if (pos == string::npos)
    return s;
  else
    return s.substr(0, pos);
}

string read_text(string fname, int maxsize = 65536) {
  char buf[maxsize];
  buf[maxsize - 1] = 0;
  ifstream stream(fname);
  stream.read(buf, maxsize - 1);
  int n = stream.gcount();
  while (n > 0 && buf[n - 1] == '\n') n--;
  return string(buf, n);
}

wstring read_text32(string fname, int maxsize = 65536) {
  char buf[maxsize];
  buf[maxsize - 1] = 0;
  ifstream stream(fname);
  stream.read(buf, maxsize - 1);
  int n = stream.gcount();
  while (n > 0 && buf[n - 1] == '\n') n--;
  return utf8_to_utf32(string(buf, n));
}

void get_codec(vector<int> &codec,
               const vector<string> &fnames,
               const wstring extra = L"") {
  set<int> codes;
  codes.insert(0);
  for (auto c : extra) codes.insert(int(c));
  for (int i = 0; i < fnames.size(); i++) {
    string fname = fnames[i];
    string base = basename(fname);
    wstring text32 = read_text32(base + ".gt.txt");
    for (auto c : text32) codes.insert(int(c));
  }
  codec.clear();
  for (auto c : codes) codec.push_back(c);
  for (int i = 1; i < codec.size(); i++) assert(codec[i] > codec[i - 1]);
}

void show(PyServer &py, Sequence &s, int subplot = 0) {
  mdarray<float> temp;
  assign(temp, s);
  if (subplot > 0) py.evalf("subplot(%d)", subplot);
  py.imshowT(temp, "cmap=cm.hot");
}

void read_lines(vector<string> &lines, string fname) {
  ifstream stream(fname);
  string line;
  lines.clear();
  while (getline(stream, line)) {
    lines.push_back(line);
  }
}

wstring separate_chars(const wstring &s, const wstring &charsep) {
  if (charsep == L"") return s;
  wstring result;
  for (int i=0; i<s.size(); i++) {
    if (i > 0) result.push_back(charsep[0]);
    result.push_back(s[i]);
  }
  return result;
}

int main1(int argc, char **argv) {
  srandomize();

  int nepoch = getienv("nepoch", 10000000);
  int save_every = getienv("save_every", 10000);
  string save_name = getsenv("save_name", "_ocr");
  int report_every = getienv("report_every", 100);
  int display_every = getienv("display_every", 0);
  int report_time = getienv("report_time", 0);
  int test_every = getienv("test_every", 10000);
  int epoch_start = getienv("epoch_start", 0);
  wstring charsep = utf8_to_utf32(getsenv("charsep", ""));
  print("*** charsep", charsep);

  if (argc < 2 || argc > 3) THROW("... training [testing]");
  vector<string> fnames, test_fnames;
  read_lines(fnames, argv[1]);
  if (argc > 2) read_lines(test_fnames, argv[2]);
  print("got", fnames.size(), "files,", test_fnames.size(), "tests");

  vector<int> codec;
  get_codec(codec, fnames, charsep);
  print("got", codec.size(), "classes");

  CLSTMOCR clstm;
  string load_name = getsenv("load", "");
  if (load_name != "") {
    clstm.load(load_name);
    print("using exist model file:",  load_name);
  }else{
    clstm.target_height = int(getrenv("target_height", 48));
    clstm.createBidi(codec, getienv("nhidden", 50));
    clstm.setLearningRate(getdenv("rate", 1e-4), getdenv("momentum", 0.9));
  }

  clstm.net->info("");

  double test_error = 9999.0;
  double best_error = 1e38;

  PyServer py;
  if (display_every > 0) py.open();

  for (int epoch = epoch_start; epoch <= nepoch; epoch++) {
    //print(epoch);
    ////////////// random select one sample  and train ////////////////////
    //int sample_id = irandom() % fnames.size();
    for (int sample_id = 0; sample_id < fnames.size(); ++sample_id){
      double start = now();
      string fname = fnames[sample_id];
      string base = basename(fname);
      wstring gt = separate_chars(read_text32(base + ".gt.txt"), charsep);
      mdarray<float> raw;
      read_png(raw, fname.c_str(), true);
      for (int i = 0; i < raw.size(); i++) raw[i] = 1 - raw[i];
      wstring pred = clstm.train(raw, gt);   // do train and get the pred output

      //////////// show the sample  /////////////
      if (display_every>0 && sample_id % display_every == 0) {
        py.evalf("clf");
        show(py, clstm.net->inputs, 411);
        show(py, clstm.net->outputs, 412);
        show(py, clstm.targets, 413);
        show(py, clstm.aligned, 414);
      }

      ////////// report the sample
      if (report_every>0 && sample_id % report_every == 0) {
        mdarray<float> temp;
        print(epoch,"-",sample_id);
        print("TRU", gt);
        print("ALN", clstm.aligned_utf8());
        print("OUT", utf32_to_utf8(pred));
        if (epoch > 0 && report_time)
          print("steptime", (now()-start) / report_every);
        //start = now();
      }

    }


    ////////////////  do validation //////////////
    if (test_fnames.size() > 0 && test_every > 0 && epoch % test_every == 0) {
      print("do test");
      double errors = 0.0;
      double count = 0.0;
      for (int test = 0; test < test_fnames.size(); test++) {
        string fname = test_fnames[test];
        string base = basename(fname);
        wstring gt = separate_chars(read_text32(base + ".gt.txt"), charsep);
        mdarray<float> raw;
        read_png(raw, fname.c_str(), true);
        for (int i = 0; i < raw.size(); i++) raw[i] = 1 - raw[i];
        wstring pred = clstm.predict(raw);
        count += gt.size();
        errors += levenshtein(pred, gt);
      }
      test_error = errors / count;
      print("TEST ERROR:", errors,"/", count,"=", test_error);
      if ( test_error < best_error) {  //save_every == 0 &&
        best_error = test_error;
        //string fname = save_name + ".clstm";
        string fname = save_name + "current-best-" + to_string(epoch) + ".clstm";
        print("got best error rate:", best_error);
        print("saving best performing network by name:", fname);
        clstm.save(fname);
      }

    }

    ///////////// save model peroidly ////////////////
    if ( save_every > 0 && epoch % save_every == 0) {
      string fname = save_name + "-" + to_string(epoch) + ".clstm";
      clstm.save(fname);
    }


  }

  return 0;
}

int main(int argc, char **argv) {
#ifdef NOEXCEPTION
  return main1(argc, argv);
#else
  try {
    return main1(argc, argv);
  } catch (const char *message) {
    cerr << "FATAL: " << message << endl;
  }
#endif
}
