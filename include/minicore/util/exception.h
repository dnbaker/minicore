#ifndef FGC_EXCEPTION_H__
#define FGC_EXCEPTION_H__
#include <stdexcept>
#include <string>
#include <iostream> // for std::cerr;

namespace minicore {

inline namespace exception {

struct verbose_runtime_error: public std::runtime_error {
    // Sometimes exception messages aren't emitted by the stdlib, so we're ensuring they are
    template<typename...A>
    verbose_runtime_error(A &&...a): std::runtime_error(std::forward<A>(a)...) {
        std::cerr << "What: " << this->what() << '\n';
    }
};

class NotImplementedError: public verbose_runtime_error {
public:
    template<typename... Args>
    NotImplementedError(Args &&...args): verbose_runtime_error(std::forward<Args>(args)...) {}

    NotImplementedError(): verbose_runtime_error("NotImplemented.") {}
};

class UnsatisfiedPreconditionError: public verbose_runtime_error {
public:
    UnsatisfiedPreconditionError(std::string msg): verbose_runtime_error(std::string("Unsatisfied precondition: ") + msg) {}

    UnsatisfiedPreconditionError(): verbose_runtime_error("Unsatisfied precondition.") {}
};

static int require(bool condition, std::string s, int ec=0) {
    if(!condition) {
        if(ec) throw verbose_runtime_error(s + " Error code: " + std::to_string(ec));
        else   throw verbose_runtime_error(s);
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

class UnsatisfiedPostconditionError: public verbose_runtime_error {
public:
    UnsatisfiedPostconditionError(std::string msg): verbose_runtime_error(std::string("Unsatisfied precondition: ") + msg) {}

    UnsatisfiedPostconditionError(): verbose_runtime_error("Unsatisfied precondition.") {}
};

static int postcondition_require(bool condition, std::string s, int ec=0) {
    if(!condition) {
        if(ec) throw UnsatisfiedPostconditionError(s + " Error code: " + std::to_string(ec));
        else throw UnsatisfiedPostconditionError(s);
    }
    return ec;
}

#ifndef MN_THROW_RUNTIME
#define MN_THROW_RUNTIME(x) do { std::cerr << x; std::exit(1);} while(0)
#endif

#ifndef PREC_REQ_EC
#define PREC_REQ_EC(condition, s, ec) \
    ::minicore::exception::precondition_require(condition, std::string(s) + '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"', ec)
#endif

#ifndef PREC_REQ
#define PREC_REQ(condition, s) PREC_REQ_EC(condition, s, 0)
#endif

#ifndef POST_REQ_EC
#define POST_REQ_EC(condition, s, ec) \
    ::minicore::exception::postcondition_require(condition, std::string(s) + '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"', ec)
#endif

#ifndef POST_REQ
#define POST_REQ(condition, s) POST_REQ_EC(condition, s, 0)
#endif


#ifndef MINOCORE_REQUIRE
#define MINOCORE_REQUIRE(condition, s) \
    ::minicore::exception::require(condition, std::string(s) + '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"')
#endif

#ifndef MINOCORE_VALIDATE
#define MINOCORE_VALIDATE(condition) \
    ::minicore::exception::validate(condition, std::string("[") + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"')
#endif

} // inline namespace exception

} // namespace minicore

#endif /* FGC_EXCEPTION_H__ */
