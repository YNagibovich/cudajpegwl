/*
	dicom_file.h
    u-pacs DICOM library

	Author : Y.Nagibovich

    ynagibovich@gmail.com
*/
#ifndef _DICOM_FILE_H_
#define _DICOM_FILE_H_

#include <stdint.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#pragma pack(push, 1)

union DCM_TAG
{
    uint32_t tag;
    uint16_t val[2];
};

struct DATA_SET
{
    DCM_TAG tag;
    uint16_t vr;
    uint16_t val_size;
};

struct DATA_SET_B
{
    DCM_TAG tag;
    uint16_t vr;
    uint16_t resrved;
    uint32_t val_size;
};

struct DATA_SET_C
{
    DCM_TAG tag;
    uint32_t val_size;
};

#pragma pack(pop)

enum VR
{
    VR_AE = 0x4541,	//16 !< Application Entity
    VR_AS = 0x5341, // 4 !< Age String
    VR_AT = 0x5441, // 4 !< Attribute Tag
    VR_CS = 0x5343, //16 !< Code String
    VR_DA = 0x4144, // 8 !< Date
    VR_DS = 0x5344, //16 !< Decimal String
    VR_DT = 0x5444, //26 !< Date Time
    VR_FD = 0x4446, // 8 !< Floating point double
    VR_FL = 0x4C46, // 4 !< Floating point single
    VR_IS = 0x5349, //12 !< Integer String
    VR_LO = 0x4f4c, //64*!< Long string
    VR_LT = 0x544c, //10240*!< Long Text
    VR_OB = 0x424f, //!< Other Byte String
    VR_OW = 0x574f, //!< Other Word String
    VR_PN = 0x4e50, //64*!< Person Name
    VR_SH = 0x4853, //16*!< Short String
    VR_SL = 0x4C53, // 4 !< Signed long
    VR_SQ = 0x5153, //***!< Sequence
    VR_SS = 0x5353, // 2 !< Signed Short
    VR_ST = 0x5453, //1024*!< Short text
    VR_TM = 0x4d54, //016!< Time
    VR_UI = 0x4955, //064!< Unique Identifier
    VR_UL = 0x4C55, // 4 !< Unsigned Long
    VR_UN = 0x4e55, //!< Unknown
    VR_US = 0x5355, // 2 !< Unsigned Short
    VR_UT = 0x5455, //!< Unlimited Text
    VR_END = 0x0,
};

typedef struct
{
    DCM_TAG tag;
    unsigned char* data_ptr;		// pointer to data
    unsigned char* item_ptr;		// pointer to this structure in array
    VR      vr;
    int     size;
    int     type;
} DCM_DATA;

typedef std::vector<DCM_DATA> tag_list;

typedef struct
{
    tag_list    tags;
    int		    seq_size;
    int8_t*	    seq_data;
}dcm_sequence_t;

typedef struct
{
    int         width;
    int         center;
    int         slope;
    int         intercept;
    int         lut;
    uint16_t    padding;
}	WL_t;

typedef struct
{
    uint16_t size;
    uint16_t first_val;
    uint16_t bpp;
} LUT_descriptor_t;

class CDicomFile
{	
public:
    CDicomFile(const char* pFilename = nullptr, uint16_t nStopGroup = 0, uint16_t nStopTag = 0);
    ~CDicomFile(void);

    // common
    bool getValue(const uint16_t nGroup, const uint16_t nTag, int& nValue);
    bool getValue(const uint16_t nGroup, const uint16_t nTag, std::string& sValue);
    bool getTag(DCM_DATA& data);
    bool getTag(const uint16_t nGroup, const uint16_t nTag, DCM_DATA& data);
    //int checkTag(const uint16_t nGroup, const uint16_t nTag, const int nValue);
    //int checkTag(const uint16_t nGroup, const uint16_t nTag, const char* sValue);
    bool isValidFile();
    size_t getImagesCount();
    unsigned char* getImageData(size_t& nImageSize, size_t nImageIdx = 0);

    // Window\Level
	bool getWL(WL_t* wlt);
	bool checkWL();

	
protected:
    bool isBigData(uint16_t vr);
    bool loadFile(const char* pFilename, const uint16_t nStopGroup = 0, const uint16_t nStopTag = 0);
    bool parseRawData(const unsigned char* pData, const int nOffset, const int nSize, const uint16_t nStopGroup = 0, const uint16_t nStopTag = 0);

    std::string                 m_sFilename;
	unsigned char*              m_pRawData;
    tag_list                    m_vTags; 
	std::vector<size_t>         m_vImageLengths;
	std::vector<unsigned char*>	m_vImagePointers;
};

#endif //_DICOM_FILE_H_