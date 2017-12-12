/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/
#pragma once

// UIDs
// qry modes
#define QRY_FIND_PATIENT			"1.2.840.10008.5.1.4.1.2.1.1" //Patient Root Query/Retrieve Information Model – FIND
#define QRY_FIND_STUDY				"1.2.840.10008.5.1.4.1.2.2.1" //Study Root Query/Retrieve Information Model – FIND
#define QRY_FIND_PAT_STUDY_ONLY		"1.2.840.10008.5.1.4.1.2.3.1" //Patient/Study Only Query/Retrieve Information Model - FIND
#define QRY_MOVE_PATIENT			"1.2.840.10008.5.1.4.1.2.1.2" //Patient Root Query/Retrieve Information Model – MOVE
#define QRY_MOVE_STUDY				"1.2.840.10008.5.1.4.1.2.2.2" //Study Root Query/Retrieve Information Model – MOVE
#define QRY_MOVE_PAT_STUDY_ONLY		"1.2.840.10008.5.1.4.1.2.3.2" //Patient/Study Only Query/Retrieve Information Model - MOVE
#define QRY_FIND_WL					"1.2.840.10008.5.1.4.31" //Modality Worklist Information Model – FIND
#define NO_QRY						"@"

// syntax
#define EXPLICIT_VR_LITTLE_ENDIAN	"1.2.840.10008.1.2.1 " //Explicit VR Little Endian
#define DCM_APPLICATION_CONTEXT		"1.2.840.10008.3.1.1.1"
#define DCM_ABSTRACT_SYNTAX			"1.2.840.10008.1.1"
#define DCM_TRANSFER_SYNTAX			"1.2.840.10008.1.2"
#define STORAGE_CT_IMAGE			"1.2.840.10008.5.1.4.1.1.2"
#define ECHO_SOP					"1.2.840.10008.1.1"
#define DCM_FIND_TSYNTAX			"1.2.840.10008.1.2.1"
#define SC_CAPTURE_IMAGE_STORAGE	"1.2.840.10008.5.1.4.1.1.7" //"Secondary Capture Image Storage"
#define SC_WSI_IMAGE_STORAGE		"1.2.840.10008.5.1.4.1.1.77.1.6" //"WSI Image Storage"

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
	VR_UT = 0x5455,  //!< Unlimited Text
	VR_END = 0x0,
};


// log modes
#define DCM_LOG_ERROR   1
#define DCM_LOG_WARNING 2
#define DCM_LOG_OK      4
#define DCM_LOG_INFO    8

#define DCM_LOG_SYSTEM_EVENT	1
#define DCM_LOG_NETWORK_EVENT	2
#define DCM_LOG_DATA_EVENT		4
#define DCM_LOG_DATA_DUMP		8

#define DCM_IMAGE_GUID_TAG "1.30"
#define DCM_STUDY_GUID_TAG "2.30"
#define DCM_SERIES_GUID_TAG "3.30"

#define DICOM_FILE_TAG			"DICM"
#define DEFAULT_IMAGE_TYPE		"ORIGINAL\\PRIMARY\\LOCALIZER\\TOP"
#define DEFAULT_CHAR_SET		"ISO_IR 100"

#ifndef MAX_PATH
#define MAX_PATH 260
#endif //MAX_PATH