#include "json_parser.hpp"
#include <sstream>
#include <cctype>
#include <stdexcept>
#include <cmath>

namespace json {

static void skip_whitespace(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(s[pos])) pos++;
}

static std::string parse_string(const std::string& s, size_t& pos) {
    if (s[pos] != '"') throw std::runtime_error("Expected '\"'");
    pos++;
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos++;
            if (s[pos] == 'n') result += '\n';
            else if (s[pos] == 't') result += '\t';
            else if (s[pos] == 'r') result += '\r';
            else if (s[pos] == '\\') result += '\\';
            else if (s[pos] == '"') result += '"';
            else result += s[pos];
        } else {
            result += s[pos];
        }
        pos++;
    }
    if (pos >= s.size()) throw std::runtime_error("Unterminated string");
    pos++;
    return result;
}

static double parse_number(const std::string& s, size_t& pos) {
    size_t start = pos;
    if (s[pos] == '-') pos++;
    while (pos < s.size() && std::isdigit(s[pos])) pos++;
    if (pos < s.size() && s[pos] == '.') {
        pos++;
        while (pos < s.size() && std::isdigit(s[pos])) pos++;
    }
    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
        pos++;
        if (s[pos] == '+' || s[pos] == '-') pos++;
        while (pos < s.size() && std::isdigit(s[pos])) pos++;
    }
    std::string num_str = s.substr(start, pos - start);
    double val = std::stod(num_str);
    if (std::isnan(val) || std::isinf(val)) {
        throw std::runtime_error("Invalid number (NaN or Inf): " + num_str);
    }
    return val;
}

static Value parse_value(const std::string& s, size_t& pos);

static Value parse_array(const std::string& s, size_t& pos) {
    if (s[pos] != '[') throw std::runtime_error("Expected '['");
    pos++;
    skip_whitespace(s, pos);
    
    std::vector<Value> arr;
    if (s[pos] != ']') {
        while (true) {
            arr.push_back(parse_value(s, pos));
            skip_whitespace(s, pos);
            if (s[pos] == ']') break;
            if (s[pos] != ',') throw std::runtime_error("Expected ',' or ']'");
            pos++;
            skip_whitespace(s, pos);
        }
    }
    pos++;
    Value result;
    result.set_array(arr);
    return result;
}

static Value parse_object(const std::string& s, size_t& pos) {
    if (s[pos] != '{') throw std::runtime_error("Expected '{'");
    pos++;
    skip_whitespace(s, pos);
    
    std::map<std::string, Value> obj;
    if (s[pos] != '}') {
        while (true) {
            skip_whitespace(s, pos);
            std::string key = parse_string(s, pos);
            skip_whitespace(s, pos);
            if (s[pos] != ':') throw std::runtime_error("Expected ':'");
            pos++;
            skip_whitespace(s, pos);
            obj[key] = parse_value(s, pos);
            skip_whitespace(s, pos);
            if (s[pos] == '}') break;
            if (s[pos] != ',') throw std::runtime_error("Expected ',' or '}'");
            pos++;
        }
    }
    pos++;
    Value result;
    result.set_object(obj);
    return result;
}

static Value parse_value(const std::string& s, size_t& pos) {
    skip_whitespace(s, pos);
    if (pos >= s.size()) throw std::runtime_error("Unexpected end of input");
    
    if (s[pos] == '"') {
        return Value(parse_string(s, pos));
    } else if (s[pos] == '[') {
        return parse_array(s, pos);
    } else if (s[pos] == '{') {
        return parse_object(s, pos);
    } else if (s[pos] == 't' && s.substr(pos, 4) == "true") {
        pos += 4;
        return Value(true);
    } else if (s[pos] == 'f' && s.substr(pos, 5) == "false") {
        pos += 5;
        return Value(false);
    } else if (s[pos] == 'n' && s.substr(pos, 4) == "null") {
        pos += 4;
        return Value();
    } else if (s[pos] == '-' || std::isdigit(s[pos])) {
        return Value(parse_number(s, pos));
    }
    throw std::runtime_error("Invalid JSON value");
}

Value parse(const std::string& json_str) {
    size_t pos = 0;
    return parse_value(json_str, pos);
}

static void stringify_value(const Value& val, std::ostringstream& out, bool compact) {
    switch (val.get_type()) {
        case Value::NULL_TYPE:
            out << "null";
            break;
        case Value::BOOL:
            out << (val.as_bool() ? "true" : "false");
            break;
        case Value::NUMBER:
            out << val.as_number();
            break;
        case Value::STRING:
            out << '"' << val.as_string() << '"';
            break;
        case Value::ARRAY: {
            out << '[';
            const auto& arr = val.as_array();
            for (size_t i = 0; i < arr.size(); i++) {
                if (i > 0) out << ',';
                stringify_value(arr[i], out, compact);
            }
            out << ']';
            break;
        }
        case Value::OBJECT: {
            out << '{';
            const auto& obj = val.as_object();
            bool first = true;
            for (const auto& kv : obj) {
                if (!first) out << ',';
                first = false;
                out << '"' << kv.first << "\":";
                stringify_value(kv.second, out, compact);
            }
            out << '}';
            break;
        }
    }
}

std::string stringify(const Value& val, bool compact) {
    std::ostringstream out;
    stringify_value(val, out, compact);
    return out.str();
}

}
