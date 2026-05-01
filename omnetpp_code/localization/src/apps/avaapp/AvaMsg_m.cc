//
// Generated file, do not edit! Created by opp_msgtool 6.2 from apps/avaapp/AvaMsg.msg.
//

// Disable warnings about unused variables, empty switch stmts, etc:
#ifdef _MSC_VER
#  pragma warning(disable:4101)
#  pragma warning(disable:4065)
#endif

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wshadow"
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wunused-parameter"
#  pragma clang diagnostic ignored "-Wc++98-compat"
#  pragma clang diagnostic ignored "-Wunreachable-code-break"
#  pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wshadow"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wold-style-cast"
#  pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#  pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif

#include <iostream>
#include <sstream>
#include <memory>
#include <type_traits>
#include "AvaMsg_m.h"

namespace omnetpp {

// Template pack/unpack rules. They are declared *after* a1l type-specific pack functions for multiple reasons.
// They are in the omnetpp namespace, to allow them to be found by argument-dependent lookup via the cCommBuffer argument

// Packing/unpacking an std::vector
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::vector<T,A>& v)
{
    int n = v.size();
    doParsimPacking(buffer, n);
    for (int i = 0; i < n; i++)
        doParsimPacking(buffer, v[i]);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::vector<T,A>& v)
{
    int n;
    doParsimUnpacking(buffer, n);
    v.resize(n);
    for (int i = 0; i < n; i++)
        doParsimUnpacking(buffer, v[i]);
}

// Packing/unpacking an std::list
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::list<T,A>& l)
{
    doParsimPacking(buffer, (int)l.size());
    for (typename std::list<T,A>::const_iterator it = l.begin(); it != l.end(); ++it)
        doParsimPacking(buffer, (T&)*it);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::list<T,A>& l)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i = 0; i < n; i++) {
        l.push_back(T());
        doParsimUnpacking(buffer, l.back());
    }
}

// Packing/unpacking an std::set
template<typename T, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::set<T,Tr,A>& s)
{
    doParsimPacking(buffer, (int)s.size());
    for (typename std::set<T,Tr,A>::const_iterator it = s.begin(); it != s.end(); ++it)
        doParsimPacking(buffer, *it);
}

template<typename T, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::set<T,Tr,A>& s)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i = 0; i < n; i++) {
        T x;
        doParsimUnpacking(buffer, x);
        s.insert(x);
    }
}

// Packing/unpacking an std::map
template<typename K, typename V, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::map<K,V,Tr,A>& m)
{
    doParsimPacking(buffer, (int)m.size());
    for (typename std::map<K,V,Tr,A>::const_iterator it = m.begin(); it != m.end(); ++it) {
        doParsimPacking(buffer, it->first);
        doParsimPacking(buffer, it->second);
    }
}

template<typename K, typename V, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::map<K,V,Tr,A>& m)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i = 0; i < n; i++) {
        K k; V v;
        doParsimUnpacking(buffer, k);
        doParsimUnpacking(buffer, v);
        m[k] = v;
    }
}

// Default pack/unpack function for arrays
template<typename T>
void doParsimArrayPacking(omnetpp::cCommBuffer *b, const T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimPacking(b, t[i]);
}

template<typename T>
void doParsimArrayUnpacking(omnetpp::cCommBuffer *b, T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimUnpacking(b, t[i]);
}

// Default rule to prevent compiler from choosing base class' doParsimPacking() function
template<typename T>
void doParsimPacking(omnetpp::cCommBuffer *, const T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimPacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

template<typename T>
void doParsimUnpacking(omnetpp::cCommBuffer *, T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimUnpacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

}  // namespace omnetpp

namespace inet {

Register_Enum(inet::AvaMessageType, (inet::AvaMessageType::POLL, inet::AvaMessageType::RESPONSE, inet::AvaMessageType::FINAL, inet::AvaMessageType::SELF_LOC));

Register_Class(AvaMsg)

AvaMsg::AvaMsg() : ::inet::FieldsChunk()
{
}

AvaMsg::AvaMsg(const AvaMsg& other) : ::inet::FieldsChunk(other)
{
    copy(other);
}

AvaMsg::~AvaMsg()
{
}

AvaMsg& AvaMsg::operator=(const AvaMsg& other)
{
    if (this == &other) return *this;
    ::inet::FieldsChunk::operator=(other);
    copy(other);
    return *this;
}

void AvaMsg::copy(const AvaMsg& other)
{
    this->SeqNo = other.SeqNo;
    this->SenderId = other.SenderId;
    this->ReceiverId = other.ReceiverId;
    this->MsgType = other.MsgType;
    this->ContenderId = other.ContenderId;
}

void AvaMsg::parsimPack(omnetpp::cCommBuffer *b) const
{
    ::inet::FieldsChunk::parsimPack(b);
    doParsimPacking(b,this->SeqNo);
    doParsimPacking(b,this->SenderId);
    doParsimPacking(b,this->ReceiverId);
    doParsimPacking(b,this->MsgType);
    doParsimPacking(b,this->ContenderId);
}

void AvaMsg::parsimUnpack(omnetpp::cCommBuffer *b)
{
    ::inet::FieldsChunk::parsimUnpack(b);
    doParsimUnpacking(b,this->SeqNo);
    doParsimUnpacking(b,this->SenderId);
    doParsimUnpacking(b,this->ReceiverId);
    doParsimUnpacking(b,this->MsgType);
    doParsimUnpacking(b,this->ContenderId);
}

int AvaMsg::getSeqNo() const
{
    return this->SeqNo;
}

void AvaMsg::setSeqNo(int SeqNo)
{
    handleChange();
    this->SeqNo = SeqNo;
}

int AvaMsg::getSenderId() const
{
    return this->SenderId;
}

void AvaMsg::setSenderId(int SenderId)
{
    handleChange();
    this->SenderId = SenderId;
}

int AvaMsg::getReceiverId() const
{
    return this->ReceiverId;
}

void AvaMsg::setReceiverId(int ReceiverId)
{
    handleChange();
    this->ReceiverId = ReceiverId;
}

AvaMessageType AvaMsg::getMsgType() const
{
    return this->MsgType;
}

void AvaMsg::setMsgType(AvaMessageType MsgType)
{
    handleChange();
    this->MsgType = MsgType;
}

int AvaMsg::getContenderId() const
{
    return this->ContenderId;
}

void AvaMsg::setContenderId(int ContenderId)
{
    handleChange();
    this->ContenderId = ContenderId;
}

class AvaMsgDescriptor : public omnetpp::cClassDescriptor
{
  private:
    mutable const char **propertyNames;
    enum FieldConstants {
        FIELD_SeqNo,
        FIELD_SenderId,
        FIELD_ReceiverId,
        FIELD_MsgType,
        FIELD_ContenderId,
    };
  public:
    AvaMsgDescriptor();
    virtual ~AvaMsgDescriptor();

    virtual bool doesSupport(omnetpp::cObject *obj) const override;
    virtual const char **getPropertyNames() const override;
    virtual const char *getProperty(const char *propertyName) const override;
    virtual int getFieldCount() const override;
    virtual const char *getFieldName(int field) const override;
    virtual int findField(const char *fieldName) const override;
    virtual unsigned int getFieldTypeFlags(int field) const override;
    virtual const char *getFieldTypeString(int field) const override;
    virtual const char **getFieldPropertyNames(int field) const override;
    virtual const char *getFieldProperty(int field, const char *propertyName) const override;
    virtual int getFieldArraySize(omnetpp::any_ptr object, int field) const override;
    virtual void setFieldArraySize(omnetpp::any_ptr object, int field, int size) const override;

    virtual const char *getFieldDynamicTypeString(omnetpp::any_ptr object, int field, int i) const override;
    virtual std::string getFieldValueAsString(omnetpp::any_ptr object, int field, int i) const override;
    virtual void setFieldValueAsString(omnetpp::any_ptr object, int field, int i, const char *value) const override;
    virtual omnetpp::cValue getFieldValue(omnetpp::any_ptr object, int field, int i) const override;
    virtual void setFieldValue(omnetpp::any_ptr object, int field, int i, const omnetpp::cValue& value) const override;

    virtual const char *getFieldStructName(int field) const override;
    virtual omnetpp::any_ptr getFieldStructValuePointer(omnetpp::any_ptr object, int field, int i) const override;
    virtual void setFieldStructValuePointer(omnetpp::any_ptr object, int field, int i, omnetpp::any_ptr ptr) const override;
};

Register_ClassDescriptor(AvaMsgDescriptor)

AvaMsgDescriptor::AvaMsgDescriptor() : omnetpp::cClassDescriptor(omnetpp::opp_typename(typeid(inet::AvaMsg)), "inet::FieldsChunk")
{
    propertyNames = nullptr;
}

AvaMsgDescriptor::~AvaMsgDescriptor()
{
    delete[] propertyNames;
}

bool AvaMsgDescriptor::doesSupport(omnetpp::cObject *obj) const
{
    return dynamic_cast<AvaMsg *>(obj)!=nullptr;
}

const char **AvaMsgDescriptor::getPropertyNames() const
{
    if (!propertyNames) {
        static const char *names[] = {  nullptr };
        omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
        const char **baseNames = base ? base->getPropertyNames() : nullptr;
        propertyNames = mergeLists(baseNames, names);
    }
    return propertyNames;
}

const char *AvaMsgDescriptor::getProperty(const char *propertyName) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    return base ? base->getProperty(propertyName) : nullptr;
}

int AvaMsgDescriptor::getFieldCount() const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    return base ? 5+base->getFieldCount() : 5;
}

unsigned int AvaMsgDescriptor::getFieldTypeFlags(int field) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldTypeFlags(field);
        field -= base->getFieldCount();
    }
    static unsigned int fieldTypeFlags[] = {
        FD_ISEDITABLE,    // FIELD_SeqNo
        FD_ISEDITABLE,    // FIELD_SenderId
        FD_ISEDITABLE,    // FIELD_ReceiverId
        0,    // FIELD_MsgType
        FD_ISEDITABLE,    // FIELD_ContenderId
    };
    return (field >= 0 && field < 5) ? fieldTypeFlags[field] : 0;
}

const char *AvaMsgDescriptor::getFieldName(int field) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldName(field);
        field -= base->getFieldCount();
    }
    static const char *fieldNames[] = {
        "SeqNo",
        "SenderId",
        "ReceiverId",
        "MsgType",
        "ContenderId",
    };
    return (field >= 0 && field < 5) ? fieldNames[field] : nullptr;
}

int AvaMsgDescriptor::findField(const char *fieldName) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    int baseIndex = base ? base->getFieldCount() : 0;
    if (strcmp(fieldName, "SeqNo") == 0) return baseIndex + 0;
    if (strcmp(fieldName, "SenderId") == 0) return baseIndex + 1;
    if (strcmp(fieldName, "ReceiverId") == 0) return baseIndex + 2;
    if (strcmp(fieldName, "MsgType") == 0) return baseIndex + 3;
    if (strcmp(fieldName, "ContenderId") == 0) return baseIndex + 4;
    return base ? base->findField(fieldName) : -1;
}

const char *AvaMsgDescriptor::getFieldTypeString(int field) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldTypeString(field);
        field -= base->getFieldCount();
    }
    static const char *fieldTypeStrings[] = {
        "int",    // FIELD_SeqNo
        "int",    // FIELD_SenderId
        "int",    // FIELD_ReceiverId
        "inet::AvaMessageType",    // FIELD_MsgType
        "int",    // FIELD_ContenderId
    };
    return (field >= 0 && field < 5) ? fieldTypeStrings[field] : nullptr;
}

const char **AvaMsgDescriptor::getFieldPropertyNames(int field) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldPropertyNames(field);
        field -= base->getFieldCount();
    }
    switch (field) {
        case FIELD_MsgType: {
            static const char *names[] = { "enum",  nullptr };
            return names;
        }
        default: return nullptr;
    }
}

const char *AvaMsgDescriptor::getFieldProperty(int field, const char *propertyName) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldProperty(field, propertyName);
        field -= base->getFieldCount();
    }
    switch (field) {
        case FIELD_MsgType:
            if (!strcmp(propertyName, "enum")) return "inet::AvaMessageType";
            return nullptr;
        default: return nullptr;
    }
}

int AvaMsgDescriptor::getFieldArraySize(omnetpp::any_ptr object, int field) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldArraySize(object, field);
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        default: return 0;
    }
}

void AvaMsgDescriptor::setFieldArraySize(omnetpp::any_ptr object, int field, int size) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount()){
            base->setFieldArraySize(object, field, size);
            return;
        }
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        default: throw omnetpp::cRuntimeError("Cannot set array size of field %d of class 'AvaMsg'", field);
    }
}

const char *AvaMsgDescriptor::getFieldDynamicTypeString(omnetpp::any_ptr object, int field, int i) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldDynamicTypeString(object,field,i);
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        default: return nullptr;
    }
}

std::string AvaMsgDescriptor::getFieldValueAsString(omnetpp::any_ptr object, int field, int i) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldValueAsString(object,field,i);
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        case FIELD_SeqNo: return long2string(pp->getSeqNo());
        case FIELD_SenderId: return long2string(pp->getSenderId());
        case FIELD_ReceiverId: return long2string(pp->getReceiverId());
        case FIELD_MsgType: return enum2string(pp->getMsgType(), "inet::AvaMessageType");
        case FIELD_ContenderId: return long2string(pp->getContenderId());
        default: return "";
    }
}

void AvaMsgDescriptor::setFieldValueAsString(omnetpp::any_ptr object, int field, int i, const char *value) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount()){
            base->setFieldValueAsString(object, field, i, value);
            return;
        }
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        case FIELD_SeqNo: pp->setSeqNo(string2long(value)); break;
        case FIELD_SenderId: pp->setSenderId(string2long(value)); break;
        case FIELD_ReceiverId: pp->setReceiverId(string2long(value)); break;
        case FIELD_ContenderId: pp->setContenderId(string2long(value)); break;
        default: throw omnetpp::cRuntimeError("Cannot set field %d of class 'AvaMsg'", field);
    }
}

omnetpp::cValue AvaMsgDescriptor::getFieldValue(omnetpp::any_ptr object, int field, int i) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldValue(object,field,i);
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        case FIELD_SeqNo: return pp->getSeqNo();
        case FIELD_SenderId: return pp->getSenderId();
        case FIELD_ReceiverId: return pp->getReceiverId();
        case FIELD_MsgType: return static_cast<int>(pp->getMsgType());
        case FIELD_ContenderId: return pp->getContenderId();
        default: throw omnetpp::cRuntimeError("Cannot return field %d of class 'AvaMsg' as cValue -- field index out of range?", field);
    }
}

void AvaMsgDescriptor::setFieldValue(omnetpp::any_ptr object, int field, int i, const omnetpp::cValue& value) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount()){
            base->setFieldValue(object, field, i, value);
            return;
        }
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        case FIELD_SeqNo: pp->setSeqNo(omnetpp::checked_int_cast<int>(value.intValue())); break;
        case FIELD_SenderId: pp->setSenderId(omnetpp::checked_int_cast<int>(value.intValue())); break;
        case FIELD_ReceiverId: pp->setReceiverId(omnetpp::checked_int_cast<int>(value.intValue())); break;
        case FIELD_ContenderId: pp->setContenderId(omnetpp::checked_int_cast<int>(value.intValue())); break;
        default: throw omnetpp::cRuntimeError("Cannot set field %d of class 'AvaMsg'", field);
    }
}

const char *AvaMsgDescriptor::getFieldStructName(int field) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldStructName(field);
        field -= base->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    };
}

omnetpp::any_ptr AvaMsgDescriptor::getFieldStructValuePointer(omnetpp::any_ptr object, int field, int i) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount())
            return base->getFieldStructValuePointer(object, field, i);
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        default: return omnetpp::any_ptr(nullptr);
    }
}

void AvaMsgDescriptor::setFieldStructValuePointer(omnetpp::any_ptr object, int field, int i, omnetpp::any_ptr ptr) const
{
    omnetpp::cClassDescriptor *base = getBaseClassDescriptor();
    if (base) {
        if (field < base->getFieldCount()){
            base->setFieldStructValuePointer(object, field, i, ptr);
            return;
        }
        field -= base->getFieldCount();
    }
    AvaMsg *pp = omnetpp::fromAnyPtr<AvaMsg>(object); (void)pp;
    switch (field) {
        default: throw omnetpp::cRuntimeError("Cannot set field %d of class 'AvaMsg'", field);
    }
}

}  // namespace inet

namespace omnetpp {

}  // namespace omnetpp

