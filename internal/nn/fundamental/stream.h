#ifndef FUNDAMENTAL_STREAM_H
#define FUNDAMENTAL_STREAM_H

#include <mutex>
#include <functional>
#include <thread>
#include <future>
#include <deque>

typedef enum{

  STREAM_FORWARD_DIR = 0,
  STREAM_BACKWARD_DIR = 1

} stream_direction_t;

template<typename xpu, index_t stream_id>
class StreamMutex{

  std::mutex _mutex;

  public:

    StreamMutex(void) {}

    std::mutex& mutex(void){
      return _mutex;
    }

};

#define STREAM_MUTEX(x, s) Context<StreamMutex<x, s>>::Instance().mutex()

template<typename xpu, index_t stream_id, typename Ret>
class StreamBase{

  using Scopelock = std::lock_guard<std::mutex>;
  using Lock = std::unique_lock<std::mutex>;

  stream_direction_t direction;

  //std::mutex mutex;
  std::unique_lock<std::mutex> _hold_on;

  std::thread thread;
  std::condition_variable cv;
  std::condition_variable cv_done;

  std::deque<std::function<void()>> q;

  std::atomic<bool> _destroy;
  std::atomic<bool> _eager;
  std::atomic<int> active;
  std::atomic<bool> _record;

  std::mutex local_mutex;

  protected:

    void _synchronize(){

      std::mutex _mutex;

      Lock lck(_mutex);

      auto lambda =  [this]()
      {
          if( this->q.size() == 0 && this->active == 0)
            return true;
          else
            return false;
      };

      cv_done.wait(lck, lambda);

    }

  public:

    StreamBase(stream_direction_t _direction):
      direction{_direction}, _eager{true}, _destroy{false},
      active{0}, _record{true}{
      thread = std::thread(StreamBase::run, (this));
    }

    ~StreamBase(void) {}

    StreamBase& eager(void){
     _eager = true;
     return *this;
    }

    StreamBase& lazy(void){
     _eager = false;
     return *this;
    }

    StreamBase& record(bool r){
      _record = r;
      return *this;
    }

    template<class F, typename ... Args>
    void put(F&& f, Args ... args){

      if (_record ){

        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared< std::packaged_task<return_type()> >(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        if (!_eager){

          if (!_hold_on)
            Scopelock lck(local_mutex);

          q.push_back([task](){(*task)();});

          cv.notify_one();

        } else{

          if (active ==0)
            Scopelock lck(STREAM_MUTEX(xpu, stream_id));

          (*task)();

       }

     }

      //return (task->get_future());

    }

    static void run(StreamBase<xpu, stream_id, Ret>* self){

      while (true){

        Lock local_lck(self->local_mutex);

        auto lambda = [&self]()
        {
          if(self->q.size() > 0 || self->_destroy)
            return true;
          else
            return false;
        };

        self->cv.wait(local_lck, lambda);

        if (self->_destroy){
          local_lck.unlock();
          break;
        }

        std::function<void()> task;

        if (self->direction == STREAM_FORWARD_DIR){ // Oldest is processed first
          task = std::move(self->q.front());
          self->q.pop_front();
        } else{ // Newest is processed firs
          task = std::move(self->q.back());
          self->q.pop_back();
        }

        local_lck.unlock();

        Lock global_lock(STREAM_MUTEX(xpu, stream_id));

        self->active++;

        if (self->_eager){
          task();
        } else{
          // ensure that any task that will be added when we run task()
          // will be excetuted directly
         self->eager(); task(); self->lazy();
        }

        self->active--;
        global_lock.unlock();
        self->cv_done.notify_all();

      }

    }

    StreamBase& hold_on(void){
      ASSERT_ERROR(!_eager, "Hold on mode can not be used with eager excecution");
      _hold_on = Lock(local_mutex);
      return *this;
    }

    StreamBase& hold_off(void){
      if ( _hold_on )
        _hold_on.unlock();
      return *this;
     }

    StreamBase& synchronize(void){
      this->_synchronize();
      return *this;
    }

};

template<typename xpu, index_t stream_id>
class Stream: public StreamBase<xpu, stream_id, void> {};

template<index_t stream_id>
class Stream<cpu, stream_id>: public StreamBase<cpu, stream_id, void>{

public:

  Stream(stream_direction_t dir): StreamBase<cpu, stream_id, void>(dir) {}

};

template<typename xpu, index_t stream_id>
class Stream2{

  std::map<stream_direction_t, Stream<xpu, stream_id> > streams;

  struct Context{

    stream_direction_t& dir;

    stream_direction_t revert_dir;

    Context(stream_direction_t& _dir, stream_direction_t _revert_dir):
            dir{_dir}, revert_dir{_revert_dir} {}

    ~Context(){ dir = revert_dir;  }

  };

  public:

    stream_direction_t dir;

    Stream2(): dir{STREAM_FORWARD_DIR} {
     streams.emplace(STREAM_FORWARD_DIR, STREAM_FORWARD_DIR);
     streams.emplace(STREAM_BACKWARD_DIR, STREAM_BACKWARD_DIR);
    }

    Context set_direction(stream_direction_t _dir){
      stream_direction_t revert_dir = this->dir;
      this->dir = _dir;
      return Context(this->dir, revert_dir);
    }

    Stream<xpu, stream_id>& get_stream(){
      DEBUG_ASSERT(streams.find(this->dir) != streams.end());
      return streams.at(this->dir);
    }

    Stream<xpu, stream_id>& get_stream(stream_direction_t _dir){
      DEBUG_ASSERT(streams.find(_dir) != streams.end());
      return streams.at(_dir);
    }

    Stream2& synchronize(){
      for (auto s : streams){
        s.second.synchronize();
      }
    }

};

#define STREAM_HANDLER(x, s) Context<Stream2<x, s>>::Instance()

#define STREAM(x, s) Context<Stream2<x, s>>::Instance().get_stream()

#define STREAM_FORWARD(x, s) Context<Stream2<x, s>>::Instance().get_stream(STREAM_FORWARD_DIR)

#define STREAM_BACKWARD(x, s) Context<Stream2<x, s>>::Instance().get_stream(STREAM_BACKWARD_DIR)

#define STREAM_CONTEXT(x, s, c) Context<Stream2<x, s>>::Instance().set_direction(c)


#ifdef TENSOR_USE_CUDA
  #include <fundamental/cuda/stream-inl.cuh>
#endif

#endif
