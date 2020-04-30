#ifndef FGC_EXCEPTION_H__
#define FGC_EXCEPTION_H__
#include <stdexcept>
#include <string>

namespace minocore {

inline namespace exception {

struct TODOError: public std::runtime_error {
    template<typename...A>
    TODOError(A &&...a): std::runtime_error(std::forward<A>(a)...) {}
};

class NotImplementedError: public std::runtime_error {
public:
    template<typename... Args>
    NotImplementedError(Args &&...args): std::runtime_error(std::forward<Args>(args)...) {}

    NotImplementedError(): std::runtime_error("NotImplemented.") {}
};

class UnsatisfiedPreconditionError: public std::runtime_error {
public:
    UnsatisfiedPreconditionError(std::string msg): std::runtime_error(std::string("Unsatisfied precondition: ") + msg) {}

    UnsatisfiedPreconditionError(): std::runtime_error("Unsatisfied precondition.") {}
};

static int require(bool condition, std::string s, int ec=0) {
    if(!condition) {
        if(ec) throw std::runtime_error(s + " Error code: " + std::to_string(ec));
        else   throw std::runtime_error(s);
    }
    return ec;
}

static int validate(bool condition, std::string s, int ec=0) {
    if(!condition) {
        if(ec) throw std::invalid_argument(s + " Error code: " + std::to_string(ec));
        else   throw std::invalid_argument(s);
    }
    return ec;
}


static int precondition_require(bool condition, std::string s, int ec=0) {
    if(!condition) {
        if(ec) throw UnsatisfiedPreconditionError(s + " Error code: " + std::to_string(ec));
        else throw UnsatisfiedPreconditionError(s);
    }
    return ec;
}

class UnsatisfiedPostconditionError: public std::runtime_error {
public:
    UnsatisfiedPostconditionError(std::string msg): std::runtime_error(std::string("Unsatisfied precondition: ") + msg) {}

    UnsatisfiedPostconditionError(): std::runtime_error("Unsatisfied precondition.") {}
};

static int postcondition_require(bool condition, std::string s, int ec=0) {
    if(!condition) {
        if(ec) throw UnsatisfiedPostconditionError(s + " Error code: " + std::to_string(ec));
        else throw UnsatisfiedPostconditionError(s);
    }
    return ec;
}

#ifndef PREC_REQ_EC
#define PREC_REQ_EC(condition, s, ec) \
    ::minocore::exception::precondition_require(condition, std::string(s) + '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"', ec)
#endif

#ifndef PREC_REQ
#define PREC_REQ(condition, s) PREC_REQ_EC(condition, s, 0)
#endif

#ifndef POST_REQ_EC
#define POST_REQ_EC(condition, s, ec) \
    ::minocore::exception::postcondition_require(condition, std::string(s) + '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"', ec)
#endif

#ifndef POST_REQ
#define POST_REQ(condition, s) POST_REQ_EC(condition, s, 0)
#endif


#ifndef MINOCORE_REQUIRE
#define MINOCORE_REQUIRE(condition, s) \
    ::minocore::exception::require(condition, std::string(s) + '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"')
#endif

#ifndef MINOCORE_VALIDATE
#define MINOCORE_VALIDATE(condition) \
    ::minocore::exception::validate(condition, '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"')
#endif

} // inline namespace exception

} // namespace minocore

#endif /* FGC_EXCEPTION_H__ */
