/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once
#include "dcm_version.h"
#include "dcm_data.h"
#include "dcm_constants.h"

// system
#include <time.h>
#include <vector>
#include <string>

typedef struct  
{	
	UINT32 tag;
	std::string data;
}DCM_QRY_item_t;

class DCM_QWERY_t
{	
public:
	DCM_QWERY_t();
	~DCM_QWERY_t();

	void Clear();

	int Size() { return v_items.size(); };

	// helpers
	std::string s_patient;
	std::string s_patient_id;
	std::string s_study_id;
	std::string s_image_id;
	std::vector<DCM_QRY_item_t> v_items;
	std::string s_mode;
};

typedef struct  
{	
	std::string PatientID;
	std::string name;
	std::string BirthDate;
	std::string Sex;
	std::string Age;
	long   GID;
	__time64_t tm_created;
	__time64_t tm_last_access;
	long state;
	void Clear();
}PATIENT_DATA_t;

typedef struct  
{	
	std::string ExamID;
	std::string name;
	std::string StudyDate;
	std::string StudyTime;
	std::string AccessionNumber;
	std::string Modality;
	std::string Description;
	std::string BodyPart;
	long   GID;
	long   PatientID;
	__time64_t tm_created;
	__time64_t tm_last_access;
	long state;
	void Clear();
}EXAM_DATA_t;

typedef struct  
{	
	std::string SeriesID;
	std::string name;
	std::string Description;
	std::string BodyPart;
	std::string Number;
	long   GID;
	long   ExamID;
	// for Viewer
	DWORD  tag;
	__time64_t tm_created;
	__time64_t tm_last_access;
	long state;
	void Clear();
}SERIES_DATA_t;

typedef struct  
{	
	std::string ImageID;
	std::string name;
	long   GID;
	long   SeriesID;
	std::string path;
	std::string SopUID;
	__time64_t tm_created;
	__time64_t tm_last_access;
	long state;
	void Clear();
}IMAGE_DATA_t;

typedef struct
{	std::string AccessionNumber;
	std::string PatID;
	std::string PatName;
	std::string PatSex;
	std::string PatBirthday;
	std::string SchProcStepSeq;
	std::string SchStartDate;
	std::string SchStartTime;
	std::string Modality;
	std::string SchAET;
	void Clear();
}DICOM_WL_t;

typedef struct  
{	
	std::string name;
	std::string PatID;
	std::string PatName;
	std::string modality;
	__time64_t qry_from;
	__time64_t qry_to;
	long mode;
	std::string AET;
	void Clear();
} DICOM_FIND_QRY_t;

typedef struct
{
	std::string name;
	std::string AET;
	std::string addr;
	uint16_t  port;
	long    save_to_arc;
	void Clear();
}DICOM_PROVIDER_t;

// storage info
typedef struct
{
	unsigned int image_id;
	FILE* datafile;
	std::string sop_UID;
	std::string image_UID;
} image_store_info_t;
