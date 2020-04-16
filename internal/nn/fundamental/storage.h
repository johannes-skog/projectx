#ifndef FUNDAMENTAL_STORAGE_H
#define FUNDAMENTAL_STORAGE_H

#include <iostream>
#include <fstream>

namespace core{

template <typename T, std::ios_base::openmode bitmask>
class Storage{};

template <typename T>
class Storage<T, std::ios::out>{

  std::fstream stream;

  public:

    Storage(void) {}

    ~Storage(void) { if (stream.is_open()) stream.close();}

    void open(const std::string filename){
      stream.open(filename, std::ios::out | std::ios::binary);
      assert(stream.is_open());
    }

    void write(T* v){
       stream.write(reinterpret_cast<const char*>(v), sizeof(T));
       assert(stream.good());
    }

    void write(T* v, const index_t N){
      stream.write(reinterpret_cast<const char*>(v), N*sizeof(T));
      assert(stream.good());
    }

};

template <typename T>
class Storage<T, std::ios::in>{

  std::fstream stream;
  std::streampos begin;
  std::streampos end;

  public:

    Storage(void): begin{0}, end{0} {}

    ~Storage(void) { if (stream.is_open()) stream.close();}

    void open(const std::string filename){
       stream.open(filename, std::ios::in | std::ios::binary);
       begin = stream.tellg();
       stream.seekg(0, std::ios::end);
       end = stream.tellg();
       stream.seekg(begin);
       assert(stream.is_open());
    }

    index_t size(void){
      assert(stream.is_open());
      return end - begin;
    }

    bool at_end(void){
       if (stream.tellg() == end) return true;
       return false;
    }

    bool check(void){ return true; }

    void read(T* v){
      stream.read(reinterpret_cast<char*>(v), sizeof(T));
      assert(stream.good());
     }

    void read(T* v, const index_t N) {
       stream.read(reinterpret_cast<char*>(v), N*sizeof(T));
       assert(stream.good());
    }

};


}

#endif
