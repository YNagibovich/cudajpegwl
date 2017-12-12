/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "dcm_version.h"
#include "dcm_constants.h"

// basic DICOM data definitions and data structures

//#ifndef int8_t
//#define int8_t unsigned char
//#endif

#pragma pack(push, 1)

union DCM_TAG
{ 
	uint32_t tag;
	uint16_t val[2];
};

union D16_TAG
{ 
	uint16_t data;
	int8_t   val[2];
};

typedef struct
{	
	int8_t context_id;
	int8_t uid[64];
	int8_t type;
	int8_t supported;
}	ASSOCIATION_t;

typedef struct
{	
	int8_t type;
	int8_t reserved;
	uint16_t length;
	char data[1];
}APPLICATION_CONTEXT_t;

typedef struct  
{	
	int8_t type;
	int8_t reserved;
	uint16_t length; // incl last transfer syntax length
	int8_t context_id; // odd int 1 to 255
	int8_t reserved1[3];
}PRESENTATION_CONTEXT_t;

typedef struct  
{	
	int8_t type;
	int8_t reserved;
	uint16_t length;
	char data[1];
}ABSTRACT_SYNTAX_t;

typedef struct  
{	
	int8_t type;
	int8_t reserved;
	uint16_t length;
	char data[1];
}TRANSFER_SYNTAX_t;

typedef struct 
{	
	int8_t type;
	int8_t resrved1;
	uint16_t reserved3;
	uint16_t length;
	uint16_t version;
	uint16_t reserved2;
	int8_t  CalledAET[16];
	int8_t  CallingAET[16];
	int8_t  reserved[32];
} ASSOCIATE_RQ_t;

typedef struct 
{	
	int8_t type;
	int8_t reserved;
	uint16_t length;
} PDU_DATA1_t;

typedef struct 
{	
	int8_t type;
	int8_t reserved1;
	uint16_t length;
	int8_t context_id;
	int8_t reserved2;
	int8_t reserved3;
	int8_t reserved4;
} PDU_DATA2_t;

typedef struct 
{	
	int8_t type;
	int8_t reserved;
	uint16_t reserved1;
	uint16_t length;
} PDU_DATA3_t;

typedef struct 
{	
	int8_t type;
	int8_t reserved;
	uint32_t length;
} PDU_DATA_TF_t;

typedef struct 
{	
	int8_t   type;
	int8_t   reserved;
	uint16_t reserved1;
	uint16_t length;
	uint16_t reserved2;
	uint16_t item_length;
	int8_t  presentation_ID;
	int8_t  tag;
} PDU_DATA_TF_EX_t;

typedef struct 
{	
	uint32_t tag;
	uint16_t val_size;
	uint16_t reserved;
	uint16_t data;
} DATA_SET_CMD_t;

// configuration
typedef struct
{
	std::string modality;
	std::string station_name;
	std::string AET;
	std::string inst_name;
	std::string inst_address;
}dicom_config_t;

typedef struct 
{
	uint32_t tag;
	VR vr;
	const char* name;
}DictEntry_t;

typedef struct
{
	DCM_TAG tag;		// tag itself
	int8_t* data_ptr;		// pointer to data
	int8_t* item_ptr;		// pointer to this structure in array
	uint16_t vr;
	int size;
	int type;
	uint16_t flags;
} DCM_DATA;

typedef std::vector<DCM_DATA> tag_list;

typedef struct
{
	tag_list tags;
	int		seq_size;
	int8_t*	seq_data;
}dcm_sequence_t;

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

struct DATA_SET_D
{
	DCM_TAG tag;
	D16_TAG val_size;
};

struct DATA_SET_E
{
	DCM_TAG tag;
	uint16_t val_size;
	uint16_t resrved;
};

struct DATA_SET_CMD
{
	uint32_t tag;
	uint16_t val_size;
	uint16_t reserved;
	uint16_t data;
};

struct PDU_DATA_TF
{
	int8_t type;
	int8_t reserved;
	uint32_t length;
};

struct PDU_DATA_TF_EX
{
	int8_t   type;
	int8_t   reserved;
	uint16_t reserved1;
	uint16_t length;
	uint16_t reserved2;
	uint16_t item_length;
	int8_t  presentation_ID;
	int8_t  tag;
};


struct PDU_DATA1
{
	int8_t type;
	int8_t reserved;
	uint16_t length;
};

struct PDU_DATA2
{
	int8_t type;
	int8_t reserved1;
	uint16_t length;
	int8_t context_id;
	int8_t reserved2;
	int8_t reserved3;
	int8_t reserved4;
};

struct ASSOCIATE_RQ
{
	int8_t type;
	int8_t resrved1;
	uint16_t reserved3;
	uint16_t length;
	uint16_t version;
	uint16_t reserved2;
	int8_t  CalledAET[16];
	int8_t  CallingAET[16];
	int8_t  reserved[32];
};

struct ASSOCIATION
{
	int8_t context_id;
	int8_t uid[64];
	int8_t type;
};

typedef struct
{
	int width;
	int center;
	int slope;
	int intercept;
	int lut;
	unsigned short padding;
}	WL_t;

typedef struct
{
	uint16_t size;
	uint16_t first_val;
	uint16_t bpp;
} LUT_descriptor_t;

#pragma pack(pop)
