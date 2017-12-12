/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once
#include <string>
#include <vector>
#include <queue>

#include "dcm_version.h"
#include "dcm_data.h"
#include "dcm_constants.h"
#include "dcm_wsi.h"
//#include "dcm_srv.h"

#define DCM_DATA_FLAG_MODIFIED	1 // item was modified
#define DCM_DATA_FLAG_ALLOCATED 2 // memory was allocated for data ptr
#define DCM_DATA_FLAG_EMPTY		0

#define DCM_SAVE_FULL	0
#define DCM_SAVE_RDCM	1

#define DCM_JPEG_NONE	0
#define DCM_JPEG_STD	1
#define DCM_JPEG_2000	2


// creation data

//Dicom-Meta-Information-Header
//Used TransferSyntax: LittleEndianExplicit
typedef struct
{
	uint16_t FileMetaInformationVersion;		//	(0002,0001) OB 00\01 
	std::string MediaStorageSOPClassUID;		//	(0002,0002) UI =CTImageStorage
	std::string	MediaStorageSOPInstanceUID;		//	(0002,0003) UI [1.3.12.2.1107.5.1.4.24661.4.0.3741314311582404] 
	std::string	TransferSyntaxUID;				//	(0002,0010) UI =LittleEndianExplicit
	std::string	ImplementationClassUID;			//	(0002,0012) UI [1.3.6.1.4.1.11157.4.5.2]
	std::string	ImplementationVersionName;		//	(0002,0013) SH [MultiTech msiCOM]
	std::string	SourceApplicationEntityTitle;	//	(0002,0016) AE [IT-PACSWS01]
	void Clear(dicom_config_t* cfg);
} dcm_creation_meta_info_t;

//Dicom-Data-Set, modality info
typedef struct
{
	std::string	SpecificCharacterSet;		//(0008,0005) CS [ISO_IR 100]                             #  10
	std::string	ImageType;					//(0008,0008) CS [ORIGINAL\PRIMARY\LOCALIZER\CT_SOM5 TOP] #  38
	std::string	SOPClassUID;				//(0008,0016) UI =CTImageStorage                          #  26
	std::string	SOPInstanceUID;				//(0008,0018) UI [1.3.12.2.1107.5.1.4.24661.4.0.37413143] #  46
	std::string	StudyDate;					//(0008,0020) DA [20031022]                               #   8
	std::string	SeriesDate;					//(0008,0021) DA [20031022]                               #   8
	std::string	AcquisitionDate;			//(0008,0022) DA [20031022]                               #   8
	std::string	ContentDate;				//(0008,0023) DA [20031022]                               #   8
	std::string	StudyTime;					//(0008,0030) TM [101921.687000]                          #  14
	std::string	SeriesTime;					//(0008,0031) TM [101926.453000]                          #  14
	std::string	AcquisitionTime;			//(0008,0032) TM [102010.579017]                          #  14
	std::string	ContentTime;				//(0008,0033) TM [102010.579017]                          #  14
	std::string	AccessionNumber;			//(0008,0050) SH (no value available)                     #   0
	std::string	Modality;					//(0008,0060) CS [CT]                                     #   2
	std::string ConversionType;				//(0008,0064) CS 
	std::string	Manufacturer;				//(0008,0070) LO [SIEMENS]                                #   8
	std::string	InstitutionName;			//(0008,0080) LO [VMedA]                                  #   6
	std::string	InstitutionAddress;			//(0008,0081) ST [Lebedeva St. Petersbu NorthWest Russia] #  42
	std::string	ReferringPhysiciansName;	//(0008,0090) PN (no value available)                     #   0
	std::string	StationName;				//(0008,1010) SH [C24661]                                 #   6
	std::string	StudyDescription;			//(0008,1030) LO [Head^HeadSeq]                           #  12
	std::string	SeriesDescription;			//(0008,103e) LO [Topogram]                               #   8
	std::string	PerformingPhysiciansName;	//(0008,1050) PN [RD]                                     #   2
	std::string	OperatorsName;				//(0008,1070) PN [PG]                                     #   2
	std::string	ManufacturersModelName;		//(0008,1090) LO [Volume Zoom]                            #  12
	//CString	ReferencedSOPClassUID;		//(0008,1150) UI =ModalityPerformedProcedureStepSOPClass  #  24
	//CString	ReferencedSOPInstanceUID;	//(0008,1155) UI [1.3.12.2.1107.5.1.4.24661.2.0.74179142] #  46
	//CString	ReferencedSOPClassUID;		//(0008,1150) UI [1.3.12.2.1107.5.9.1]                    #  20
	// WSI
	std::string	VolumetricProperies;		//(0008,9206) CS [VOLUME]								  #  12
	void Clear(dicom_config_t* cfg);
} dcm_creation_modality_info_t;

//Dicom-Data-Set, patient info
typedef struct
{
	std::string	PatientsName;		//(0010,0010) PN [KOTOVA A.V.]                            #  12
	std::string	PatientID;			//(0010,0020) LO [1121/2/SH]                              #  10
	std::string	PatientsBirthDate;	//(0010,0030) DA [19861215]                               #   8
	std::string	PatientsSex;		//(0010,0040) CS [F]                                      #   2
	std::string	PatientsAge;		//(0010,1010) AS [016Y]                                   #   4
	std::string	PatientComments;	//(0010,4000) LT 
	void Clear();
} dcm_creation_patient_info_t;

//Dicom-Data-Set, body part info
typedef struct
{
	std::string	BodyPartExamined;	//(0018,0015) CS [HEAD]                                   #   4
	std::string	SoftwareVersions;	//(0018,1020) LO [VA40C]                                  #   6
	std::string	ProtocolName;		//(0018,1030) LO [HeadSeq]                                #   8
	void Clear();
} dcm_creation_bodypart_info_t;

//Dicom-Data-Set, exam info


//Dicom-Data-Set, exam info
typedef struct
{
	std::string	StudyInstanceUID;	//(0020,000d) UI [1.3.12.2.1107.5.1.4.24661.4.0.3737817411869118] #  46
	std::string	SeriesInstanceUID;	//(0020,000e) UI [1.3.12.2.1107.5.1.4.24661.4.0.3737817829992249] #  46
	std::string	StudyID;			//(0020,0010) SH [1]											   #   2
	std::string	SeriesNumber;		//(0020,0011) IS [1]											   #   2
	std::string	AcquisitionNumber;	//(0020,0012) IS [1]											   #   2
	std::string	InstanceNumber;		//(0020,0013) IS [1]											   #   2
	std::string PatientOrientation; //(0020,0020) CS 
	// WSI
	std::string FrameofReferenceUID;		//(0020,0052) UI 
	std::string PositionReferenceIndicator; //(0020,1040) LO
	void* DimensionOrganizationSequence;	//(0020,9221) SQ
	void* DimensionIndexSequence;			//(0020,9222) SQ
	void Clear(dicom_config_t* cfg);
} dcm_creation_exam_info_t;

//Dicom-Data-Set, image info
typedef struct
{
	uint16_t SamplesPerPixel;					//(0028,0002) US 1                                        #   2
	std::string PhotometricInterpretation;		//(0028,0004) CS [MONOCHROME2]                            #  12
	uint16_t PlanarConfiguration;				//(0028,0006) US 0                                        #   2
	std::string NumberOfFrames;				//(0028,0008) IS [1]                                        #   2
	uint16_t Rows;							//(0028,0010) US 512                                      #   2
	uint16_t Columns;							//(0028,0011) US 512                                      #   2
	std::string PixelSpacing;					//(0028,0030) DS [1\1]                                    #   4
	uint16_t BitsAllocated;					//(0028,0100) US 16                                       #   2
	uint16_t BitsStored;						//(0028,0101) US 12                                       #   2
	uint16_t HighBit;							//(0028,0102) US 11                                       #   2
	uint16_t PixelRepresentation;				//(0028,0103) US 0                                        #   2
	std::string BurnedInAnnotation;			//(0028,0301) CS 	NO
	std::string WindowCenter;					//(0028,1050) DS [50\50]                                  #   6
	std::string WindowWidth;					//(0028,1051) DS [500\350]                                #   8
	std::string RescaleIntercept;				//(0028,1052) DS [-1024]                                  #   6
	std::string RescaleSlope;					//(0028,1053) DS [1]                                      #   2
	std::string WindowCenterWidthExplanation;	//(0028,1055) LO [WINDOW1\WINDOW2]                        #  16
	std::string LossyImageCompression;			//(0028,2110) CS 	1 	
	std::string LossyImageCompressionRatio;		//(0028,2112) DS 	1-n 	
	std::string LossyImageCompressionMethod;	//(0028,2114) CS
	// Dicom-Data-Set, image data
	void* PixelData;						//(7fe0,0010) OW 0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000\0000... # 524288
	dcm_tiles_list tiles;
	int DataSetTrailingPadding;				//(fffc,fffc) OB 00\00                                    #   2
	int image_length;
	void Clear(bool bCleanImage);
	int image_type;
	int image_quality;

#ifdef USE_XIMAGE	
	BOOL SetImage(CxImage* image, int jpeg_type, int jpeg_qty);
	CxImage* cImage;
#endif // USE_XIMAGE	
	bool SetImageInfo(int w, int h, int bpp, int framesnumber);
} dcm_creation_image_info_t;

//Dicom-Data-Set,study info
typedef struct
{
	std::string RequestedProcedureDescription;	//(0032,1060) LO [Head HeadSeq]                           #  12
	std::string StudyComments;					//(0032,4000) LT (no value available)                     #   0
	void Clear();
} dcm_creation_study_info_t;

typedef struct 
{
	// modality info
	std::string	ReferringPhysiciansName;	//(0008,0090) PN 64
	std::string	StudyDescription;			//(0008,1030) LO 64
	std::string	SeriesDescription;			//(0008,103e) LO 
	std::string	PerformingPhysiciansName;	//(0008,1050) PN 
	std::string	OperatorsName;				//(0008,1070) PN 
	//patient info
	std::string	PatientsName;		//(0010,0010) PN 
	std::string	PatientID;			//(0010,0020) LO 
	std::string	PatientsBirthDate;	//(0010,0030) DA [19861215]           
	std::string	PatientsSex;		//(0010,0040) CS [F] 16                 
	std::string	PatientsAge;		//(0010,1010) AS [016Y] 4    
	std::string	PatientComments;	//(0010,4000) LT
	//body part info
	std::string	BodyPartExamined;	//(0018,0015) CS [HEAD]     16                              
	//exam info
	std::string	StudyID;			//(0020,0010) SH 16
	std::string	SeriesNumber;		//(0020,0011) IS 12
	std::string	AcquisitionNumber;	//(0020,0012) IS 12
	std::string	InstanceNumber;		//(0020,0013) IS 12
	//study info
	std::string StudyComments;		//(0032,4000) LT 10240
	void Clear();
	// aux
	std::string filepath;
	int image_type;
	int partner_id;
	bool is_single;
} dcm_creation_info_t;


class dcm_creation_t
{
public:
	dcm_creation_meta_info_t meta_info;
	dcm_creation_modality_info_t modality_info;
	dcm_creation_patient_info_t patient_info;
	dcm_creation_bodypart_info_t bodypart_info;
	dcm_creation_exam_info_t exam_info;
	dcm_creation_image_info_t image_info;
	dcm_creation_study_info_t study_info;
	dcm_WSI_info_t wsi_info;
	void Clear(dicom_config_t* cfg);
	void Set(dcm_creation_info_t* info);
	void SetWSIMode();
	dcm_creation_t(dicom_config_t* cfg);
	~dcm_creation_t(void);
	bool m_bWSImode;
private:
	dicom_config_t* dcm_cfg;
};

class CDICOMFile
{	
public:
	// Window-Level
	int8_t* GetImageWL(WL_t* wlt, int image_id=0);
	bool GetWL(WL_t* wlt);
	bool HasWL();
	bool GetLUT(int8_t* lut, LUT_descriptor_t* lut_descr);
	// image specific
	int GetDCMCompression( int* is_lossless=NULL);
	int GetIDX() { return IDX;};
	int GetDCMbpp();
	int GetDCM_calib_x();
	int GetDCM_calib_y();
	uint32_t GetImagesCount();
	char* GetDCMCompressionName();
	uint64_t GetImageLength( int image_id=0);
	int8_t* GetImagePtr( int image_id=0);
	// common
	CDICOMFile(void);
	~CDICOMFile(void);
	bool Init();
	// tag manipulation
	char* GetDCMItemTag_T(int idx);
	char* GetDCMItemVal_T(int idx);
	char* GetDCMItemVR_T(int idx);
	int GetItemLength(int idx);
	char* GetDCMItemDescription(int idx);
	VR GetItemVR(uint32_t tag);
	char* GetDCMItemEx(int tag, int next=0);
	// data item manipulation
	int GetDCMItemCout() { return DCM_cnt; };
	bool GetDCMItem(int tag, uint16_t *data, int next=0);
	bool GetDCMItem(int tag, int16_t *data, int next = 0);
	bool GetDCMItem(int tag, uint32_t *data, int next=0);
	bool GetDCMItem(int tag, int32_t *data, int next=0);
	bool GetDCMItem(int tag, float *data, int next=0);
	bool GetDCMItem(int tag, double *data, int next=0);
	bool GetDCMItem(int tag, int8_t **data, int next=0);
	bool SetDCMItem(int tag, uint16_t *data);
	bool SetDCMItem(int tag, int16_t *data);
	bool SetDCMItem(int tag, void* data, int datalen);
	bool SetDCMItem(int tag, uint32_t *data);
	bool SetDCMItem(int tag, int32_t *data);
	bool SetDCMItem(int tag, float *data);
	bool SetDCMItem(int tag, double *data);
	bool SetDCMItem(int tag, int8_t **data);
	bool SetDCMItem(int tag, char *data);
	bool AddDCMItem(int tag, uint16_t *data, tag_list& dcmtags);
	bool AddDCMItem(int tag, int16_t *data, tag_list& dcmtags);
	bool AddDCMItem(int tag, void* data, int datalen, tag_list& dcmtags);
	bool AddDCMItem(int tag, uint32_t *data, tag_list& dcmtags);
	bool AddDCMItem(int tag, int32_t *data, tag_list& dcmtags);
	bool AddDCMItem(int tag, float *data, tag_list& dcmtags);
	bool AddDCMItem(int tag, double *data, tag_list& dcmtags);
	bool AddDCMItem(int tag, int8_t **data, tag_list& dcmtags);
	bool AddDCMItem(int tag, const char *data, tag_list& dcmtags);
	// dicom file
	bool Convert2Send(char* buffer, int max_len);
	bool Convert2Send(char* path);
	FILE* Convert2SendEx(char* path);
	bool CloseDICOMFile();
	bool IsDICOMDir();
	bool OpenDICOMFile(char* filename, int mode = 0);
	bool CreateDICOMFile(dcm_creation_t* ctx, char* src_file, int nQuality); // 0-100
	bool SaveDICOMFile(char* filename, int mode = DCM_SAVE_FULL);

	bool AddTiles(dcm_tiles_list* info);
	bool CreateSequence(dcm_sequence_t& info, bool bAlt = false);
	int CalcSeqLen(dcm_sequence_t& info, int nStart);
	int CalcSeqLen(dcm_sequence_t& info);
	bool CreateDICOMFile(dcm_creation_t* ctx, bool bIsWSI);
	int SaveTag(int tag, int8_t* dst, tag_list& dcmtags, bool bAsIDX = false);
	// aux
	void DICOMTimerProc();
	int GetIntVal(char* ptr);
	std::string file_name;

#ifdef USE_XIMAGE
	BOOL LoadImage(CxImage* image, int image_id = 0);
	BOOL CreateDICOMFile(dcm_creation_t* ctx, CxImage* image);
	BOOL SetWL(WL_t* wlt, CxImage* image);
#endif // USE_XIMAGE

protected:
	bool IsBigData(uint16_t vr);
	bool SaveTags(int tag, FILE* outfile);
	bool SaveTag(int tag, FILE* outfile);
	bool SaveTag(int tag, FILE* outfile, tag_list& _items);
	int FindTags(int tag, int* tags, uint32_t* total_len, tag_list& dcmtags);
	FILE* dicom_file;
	int IDX;
	std::string compression_name;
	int8_t* dicom_data_buffer;
	int8_t* WL_BUF;
	int8_t* buf_command;
	uint64_t  buf_command_size;
	uint64_t  dicom_file_size;
	uint64_t  dicom_buffer_size;
	bool LoadFile(int mode=0);
	// data operations
	void InitCommand();
	tag_list m_items; // tags
	int DCM_cnt;
	int8_t* DCM_CMD;
	uint32_t dcm_cmd_length;
	int FindItemIdx(int tag, int startfrom=0);
	int FindItemIdx(int tag, tag_list& dcmtags);
	int8_t* FillDCMItem(int& idx, int tag, int len, tag_list& dcmtags);
private:
	char m_string[MAX_PATH];
	int8_t* DIR_ptr;
	int status;
	int dcm_data_size;
	int x_len;
	uint16_t DICOM_COMMAND;
	ASSOCIATION association[128];
	int8_t   association_PID;
	uint16_t association_size;
	uint16_t association_part;
	uint32_t dicom_timer_cnt;
	// C-STORE process vaiables
	uint16_t store_message_id;
	uint16_t store_pdv_size;
	int8_t   store_sop_uid[64];
	int8_t   store_instance_uid[64];
	std::string store_file_name;
	bool last_data;
	bool is_explicit;
	// image data support
	std::vector<uint64_t> v_Image_lengths;
	std::vector<void*>	v_Image_pointers;
};

