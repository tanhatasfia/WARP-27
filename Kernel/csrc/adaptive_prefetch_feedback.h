
#pragma once
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cb_sm_prefetch {
// (cuda device, bits, projection mode, K, V, SplitK, CTAs, slice tiles)
using Key = std::tuple<int,int,int,int,int,int,int,int>;
class Selector {
    std::mutex mu_;
    std::string loaded_path_;
    std::map<Key,int> choices_;
    void load(const std::string& path) {
        std::ifstream in(path);
        if(!in.good()) throw std::runtime_error("Cannot open WARP_SM_PROFILE: " + path);
        std::map<Key,int> fresh;
        std::string line;
        int line_no=0;
        while(std::getline(in,line)) {
            ++line_no;
            if(line.empty() || line[0]=='#') continue;
            std::istringstream ss(line);
            int dev,bits,mode,k,v,split,blocks,slice,dist;
            if(!(ss>>dev>>bits>>mode>>k>>v>>split>>blocks>>slice>>dist) || dist<1 || dist>3)
                throw std::runtime_error("Invalid SM profile line " + std::to_string(line_no));
            fresh[std::make_tuple(dev,bits,mode,k,v,split,blocks,slice)]=dist;
        }
        choices_.swap(fresh);
        loaded_path_=path;
    }
public:
    static Selector& instance() {static Selector s; return s;}
    int select(const Key& key,int max_distance) {
        if(max_distance<=0)return 0;
        const char* path=std::getenv("WARP_SM_PROFILE");
        if(!path || !*path) throw std::runtime_error(
            "SM-active adaptive prefetch requires WARP_SM_PROFILE. "
            "Calibrate with scripts/profile_sm_prefetch.py using ncu, or set "
            "WARP_MARLIN_PREFETCH=1..3 for fixed mode.");
        std::lock_guard<std::mutex> guard(mu_);
        if(loaded_path_!=path)load(path);
        auto it=choices_.find(key);
        if(it==choices_.end()) {
            // No silent heuristic: fail with an explicit shape key.
            std::ostringstream msg;
            msg<<"Missing SM-active profile entry for";
            std::apply([&](auto... fields){((msg<<' '<<fields),...);},key);
            msg<<". Run hardware-counter calibration for this launch shape.";
            throw std::runtime_error(msg.str());
        }
        return std::min(it->second,max_distance);
    }
};
} 
