#pragma once
#include <string>
#include <vector>
#include <map>

namespace json {
    class Value {
    public:
        enum Type { NULL_TYPE, BOOL, NUMBER, STRING, ARRAY, OBJECT };
        
        Value() : type(NULL_TYPE) {}
        Value(int i) : type(NUMBER), num(static_cast<double>(i)) {}
        Value(double d) : type(NUMBER), num(d) {}
        Value(const std::string& s) : type(STRING), str(s) {}
        Value(bool b) : type(BOOL), boolean(b) {}
        
        Type get_type() const { return type; }
        double as_number() const { return num; }
        std::string as_string() const { return str; }
        bool as_bool() const { return boolean; }
        const std::vector<Value>& as_array() const { return arr; }
        const std::map<std::string, Value>& as_object() const { return obj; }
        
        void set_array(const std::vector<Value>& a) { type = ARRAY; arr = a; }
        void set_object(const std::map<std::string, Value>& o) { type = OBJECT; obj = o; }
        
        Value& operator[](const std::string& key) { return obj[key]; }
        const Value& operator[](const std::string& key) const { 
            auto it = obj.find(key);
            static Value null_val;
            return it != obj.end() ? it->second : null_val;
        }
        
        Value& operator[](size_t idx) { return arr[idx]; }
        const Value& operator[](size_t idx) const { return arr[idx]; }
        
        size_t size() const { return type == ARRAY ? arr.size() : obj.size(); }
        bool has(const std::string& key) const { return obj.find(key) != obj.end(); }
        
    private:
        Type type;
        double num = 0;
        std::string str;
        bool boolean = false;
        std::vector<Value> arr;
        std::map<std::string, Value> obj;
    };
    
    Value parse(const std::string& json_str);
    std::string stringify(const Value& val, bool compact = true);
}
