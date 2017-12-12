/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once
#include <windows.h>
#include "dcm_version.h"
#include "dcm_data.h"
#include "dcm_wsi.h"
#include "dcm_qry.h"
#include "dcm_utils.h"
#include "dcm_file.h"

#include <tchar.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <direct.h>


// swap
void sb(uint16_t *d)
{
	D16_TAG data;

	data.data = *d;
	int8_t b = data.val[0];
	data.val[0] = data.val[1];
	data.val[1] = b;
	*d = data.data;
}

uint32_t sw(uint32_t d)
{
	DCM_TAG t;
	uint16_t v;

	t.tag = d;
	v = t.val[0];
	t.val[0] = t.val[1];
	t.val[1] = v;
	return t.tag;
}

//
std::wstring s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}

static const std::string WHITESPACE = " \n\r\t";

static std::string TrimLeft(const std::string& s)
{
	size_t startpos = s.find_first_not_of(WHITESPACE);
	return (startpos == std::string::npos) ? "" : s.substr(startpos);
}

static std::string TrimRight(const std::string& s)
{
	size_t endpos = s.find_last_not_of(WHITESPACE);
	return (endpos == std::string::npos) ? "" : s.substr(0, endpos + 1);
}

std::string _trim(const std::string& s)
{
	std::string ret;

	ret = TrimRight(TrimLeft(s));
	ret.shrink_to_fit();
	return ret;
}



// date and time
static const int cBUF_SIZE = 128;

std::string dcm_utl_get_date()
{
	std::time_t rawtime;
	std::tm* timeinfo;
	char buffer[cBUF_SIZE];
	std::string ret;

	std::time(&rawtime);
	timeinfo = std::localtime(&rawtime);

	std::strftime(buffer, cBUF_SIZE, "%Y%m%d", timeinfo);
	ret = buffer;
	return ret;
}

std::string dcm_utl_get_time()
{
	std::time_t rawtime;
	std::tm* timeinfo;
	char buffer[cBUF_SIZE];
	std::string ret;

	std::time(&rawtime);
	timeinfo = std::localtime(&rawtime);

	std::strftime(buffer, cBUF_SIZE, "%H%M%S.000000", timeinfo);
	ret = buffer;
	return ret;
}

std::string dcm_utl_get_datetime()
{
	std::time_t rawtime;
	std::tm* timeinfo;
	char buffer[cBUF_SIZE];
	std::string ret;

	std::time(&rawtime);
	timeinfo = std::localtime(&rawtime);

	std::strftime(buffer, cBUF_SIZE, "%Y%m%d%H%M%S", timeinfo);
	ret = buffer;
	return ret;
}


// creation structs

void dcm_creation_meta_info_t::Clear(dicom_config_t* cfg)
{
	FileMetaInformationVersion = 0x0100;
	MediaStorageSOPClassUID = GetModalitySOPClassUID(cfg->modality.c_str());
	MediaStorageSOPInstanceUID = DCM_Create_GUID(cfg->station_name.c_str(), DCM_IMAGE_GUID_TAG);
	TransferSyntaxUID = EXPLICIT_VR_LITTLE_ENDIAN;
	ImplementationClassUID = IMPLEMENTATION_CLASS_UID;
	ImplementationVersionName = DICOM_IMPLEMENTATION_NAME;
	SourceApplicationEntityTitle = cfg->AET;
}

void dcm_creation_modality_info_t::Clear(dicom_config_t* cfg)
{
	std::string s_date = dcm_utl_get_date();
	std::string s_time = dcm_utl_get_time();

	SpecificCharacterSet = DEFAULT_CHAR_SET;
	ImageType = DEFAULT_IMAGE_TYPE;
	SOPClassUID = GetModalitySOPClassUID(cfg->modality.c_str());
	SOPInstanceUID = DCM_Create_GUID(cfg->station_name.c_str(), DCM_IMAGE_GUID_TAG);

	StudyDate = s_date;
	SeriesDate = s_date;
	AcquisitionDate = s_date;
	ContentDate = s_date;

	StudyTime = s_time;
	SeriesTime = s_time;
	AcquisitionTime = s_time;
	ContentTime = s_time;

	AccessionNumber.clear();
	Modality = cfg->modality;
	ConversionType = "SI";	//DV = Digitized Video
	//DI = Digital Interface
	//DF = Digitized Film
	//WSD = Workstation
	//SD = Scanned Document
	//SI = Scanned Image
	//DRW = Drawing
	//SYN = Synthetic Image
	Manufacturer = MANUFACTURER_NAME;
	InstitutionName = cfg->inst_name;
	InstitutionAddress = cfg->inst_address;
	ReferringPhysiciansName.clear();
	StationName = cfg->station_name;
	StudyDescription.clear();
	SeriesDescription.clear();
	PerformingPhysiciansName.clear();
	OperatorsName.clear();
	ManufacturersModelName = MANUFACTURER_MODEL_NAME;
	VolumetricProperies = "VOLUME";
};

void dcm_creation_exam_info_t::Clear(dicom_config_t* cfg)
{
	StudyInstanceUID = DCM_Create_GUID(cfg->station_name.c_str(), DCM_STUDY_GUID_TAG);
	SeriesInstanceUID = DCM_Create_GUID(cfg->station_name.c_str(), DCM_SERIES_GUID_TAG);
	StudyID = "1";
	SeriesNumber = "1";
	AcquisitionNumber = "1";
	InstanceNumber = "1";
	PatientOrientation.clear();
	FrameofReferenceUID.clear();
	PositionReferenceIndicator.clear();
	DimensionOrganizationSequence = NULL;
	DimensionIndexSequence = NULL;
}

void dcm_creation_study_info_t::Clear()
{
	RequestedProcedureDescription.clear();
	StudyComments.clear();
}

void dcm_creation_image_info_t::Clear(bool bCleanImage)
{
	SamplesPerPixel = 1;
	PhotometricInterpretation = "RGB";
	PlanarConfiguration = 0;
	Rows = 0;
	Columns = 0;
	PixelSpacing = "1\\1";
	BitsAllocated = 8;
	BitsStored = 8;
	HighBit = 7;
	PixelRepresentation = 0;
	WindowCenter = "50\\50";
	WindowWidth = "500\\500";
	RescaleIntercept = "-1024";
	RescaleSlope = "1";
	WindowCenterWidthExplanation = "WINDOW1\\WINDOW2";
	PixelData = NULL;
	DataSetTrailingPadding = 0;
	image_length = 0;
	LossyImageCompression.clear();
	LossyImageCompressionRatio.clear();
	LossyImageCompressionMethod.clear();
	image_type = DCM_JPEG_NONE;
	image_quality = 100;
	NumberOfFrames = "1";
#ifdef USE_XIMAGE
	if (bCleanImage)
		cImage = NULL;
#endif // USE_XIMAGE
	tiles.clear();
}

void dcm_creation_t::SetWSIMode()
{
	m_bWSImode = true;
	meta_info.MediaStorageSOPClassUID = SC_WSI_IMAGE_STORAGE;
	// we will use RGB in this DEMO
	meta_info.TransferSyntaxUID = "1.2.840.10008.1.2.1"; // explicit VR little endian
	modality_info.SOPClassUID = SC_WSI_IMAGE_STORAGE;
	modality_info.VolumetricProperies = "VOLUME";
	modality_info.ImageType = "DERIVED\\PRIMARY\\VOLUME\\NONE";
	exam_info.PositionReferenceIndicator = "SLIDE_CORNER";
	exam_info.FrameofReferenceUID = "0000"; // TBD ref image UID
	image_info.BurnedInAnnotation = "NO";
}

void dcm_creation_info_t::Clear()
{
	ReferringPhysiciansName.clear();
	StudyDescription.clear();
	SeriesDescription.clear();
	PerformingPhysiciansName.clear();
	OperatorsName.clear();
	PatientsName.clear();
	PatientID.clear();
	PatientsBirthDate.clear();
	PatientsSex.clear();
	PatientsAge.clear();
	BodyPartExamined.clear();
	StudyID = "1";
	SeriesNumber = "1";
	AcquisitionNumber = "1";
	InstanceNumber = "1";
	StudyComments.clear();
	filepath.clear();
	PatientComments.clear();
	image_type = DCM_JPEG_NONE;
	is_single = true;
	partner_id = -1;
}

void dcm_creation_t::Set(dcm_creation_info_t* info)
{
	if (info == NULL) return;
	// modality info
	modality_info.ReferringPhysiciansName = info->ReferringPhysiciansName;
	modality_info.StudyDescription = info->StudyDescription;
	modality_info.SeriesDescription = info->SeriesDescription;
	modality_info.PerformingPhysiciansName = info->PerformingPhysiciansName;
	modality_info.OperatorsName = info->OperatorsName;
	//patient info
	patient_info.PatientsName = info->PatientsName;
	patient_info.PatientID = info->PatientID;
	patient_info.PatientsBirthDate = info->PatientsBirthDate;
	patient_info.PatientsSex = info->PatientsSex;
	patient_info.PatientsAge = info->PatientsAge;
	patient_info.PatientComments = info->PatientComments;
	//body part info
	bodypart_info.BodyPartExamined = info->BodyPartExamined;
	//exam info
	exam_info.StudyID = info->StudyID;
	exam_info.SeriesNumber = info->SeriesNumber;
	exam_info.AcquisitionNumber = info->AcquisitionNumber;
	exam_info.InstanceNumber = info->InstanceNumber;
	//study info
	study_info.StudyComments = info->StudyComments;
}

void dcm_WSI_info_t::Clear()
{
	ContainerIdentifier.clear(); // (0040, 0512) LO 1
	IssuerOfTheContainerIdentifierSequence = NULL;// (0040, 0513) SQ  1
	AlternateContainerIdentifierSequence = NULL;// (0040, 0515) SQ  1
	ContainerTypeCodeSequence = NULL;//(0040, 0518) SQ
	AcquisitionContextSequence = NULL; //(0040, 0555) SQ
	SpecimenContextSequence = NULL;// (0040, 0560) SQ

	ImagedVolumeWidth = 1; // (0048, 0001) FL 1
	ImagedVolumeHeight = 1; // (0048, 0002) FL 1
	ImagedVolumeDepth = 1; //  (0048, 0003) FL 1

	TotalPixelMatrixColumns = 0; // (0048, 0006) UL 1
	TotalPixelMatrixRows = 0; // (0048, 0007) UL 1
	TotalPixelMatrixOriginSequence = NULL; // (0048, 0008) SQ 1
	SpecimenLabelInImage.clear(); // (0048, 0010) CS 1
	FocusMethod = "MANUAL"; // (0048, 0011) CS  1
	ExtendedDepthOfField = "NO"; // (0048, 0012) CS 1
	NumberOfFocalPlanes = 1; //  (0048, 0013) US 1
	DistanceBetweenFocalPlanes = 1; // (0048, 0014) FL 1
	RecommendedAbsentPixelCIELabValue = 0; //  US 3
	IlluminatorTypeCodeSequence = NULL; // (0048, 0100) SQ 1
	ImageOrientationSlide = "0\\-1\\0\\-1\\0\\0"; // (0048, 0102) DS 6
	OpticalPathSequence = NULL; // (0048, 0105) SQ 1
	SharedFunctionalGroupsSequence = NULL;  // (5200, 9229) SQ     1
	PerFrameFunctionalGroupsSequence = NULL; // (5200, 9230) SQ    1
	SpecimenIdentifier.clear(); //	0x00400551, VR_LO, "",
	SpecimenUID.clear(); //	0x00400554, VR_UI, "",
	XOffsetinSlideCoordinateSystem.clear();
	YOffsetinSlideCoordinateSystem.clear();
	ZOffsetinSlideCoordinateSystem.clear();
}

// creation structs

void dcm_creation_patient_info_t::Clear()
{
	PatientsName.clear();
	PatientID.clear();
	PatientsBirthDate.clear();
	PatientsSex.clear();
	PatientsAge.clear();
	PatientComments.clear();
}

void dcm_creation_bodypart_info_t::Clear()
{
	BodyPartExamined.clear();
	SoftwareVersions = DICOM_LIB_VERSION;
	ProtocolName.clear();
}

dcm_creation_t::dcm_creation_t(dicom_config_t* cfg)
{
	dcm_cfg = cfg;
	Clear(cfg);
	m_bWSImode = false;
}

dcm_creation_t::~dcm_creation_t()
{
	Clear(dcm_cfg);
}

bool dcm_creation_image_info_t::SetImageInfo(int w, int h, int bpp, int franesnumber)
{
	bool retval = true;
	char ttt[32];

	Clear(false);
	PhotometricInterpretation = "RGB";
	SamplesPerPixel = 3; // for RGB
	Rows = h;
	Columns = w;
	BitsAllocated = bpp;
	BitsStored = bpp;
	HighBit = BitsStored - 1;
	PixelRepresentation = 0;
	image_type = DCM_JPEG_NONE;
	image_quality = 100;
	LossyImageCompression = "00";
	PixelData = NULL;
	image_length = BitsAllocated*Rows*Columns;
	LossyImageCompressionRatio = "0";
	NumberOfFrames = _itoa(franesnumber, ttt, 10);
	return retval;
}

void dcm_creation_t::Clear(dicom_config_t* cfg)
{
	meta_info.Clear(cfg);
	modality_info.Clear(cfg);
	patient_info.Clear();
	bodypart_info.Clear();
	exam_info.Clear(cfg);
	image_info.Clear(true);
	study_info.Clear();
	wsi_info.Clear();
	if (image_info.image_type != DCM_JPEG_NONE)
	{
		if (image_info.PixelData)
			free(image_info.PixelData);
	}
}

void DICOM_PROVIDER_t::Clear()
{
	name.clear();
	AET.clear();
	addr.clear();
	port = 104;
	save_to_arc = 0;
}

void DCM_QWERY_t::Clear()
{
	unsigned int i;

	s_mode.clear();
	for (i = 0; i < v_items.size(); i++)
	{
		v_items[i].data.clear();
		v_items[i].tag = 0;
	}
	v_items.clear();
	s_patient.clear();
	s_patient_id.clear();
	s_study_id.clear();
	s_image_id.clear();
}

// AUX
std::string GetTempFile()
{
	TCHAR temp_path[MAX_PATH];
	TCHAR buffer[MAX_PATH];
	std::string ret;

	// create storage
	GetTempPath(MAX_PATH, temp_path);
	if (GetTempFileName(temp_path, _T("UPA"), 0, buffer))
	{
		std::wstring wStr = buffer;
		ret = std::string(wStr.begin(), wStr.end());
	}
	else 
		ret.clear();
	return ret;
}

// aux
std::string GetModalitySOPClassUID(const char* modality)
{
	std::string retval;

	if (_strcmpi(modality, "SM"))
		retval = SC_WSI_IMAGE_STORAGE;
	else
		retval = SC_CAPTURE_IMAGE_STORAGE;
	return retval;
}

std::string DCM_Create_GUID(const char* part1, char* part2)
{
	std::string retval, s;
	int z;

	retval = OWN_GUID;
	if (part1 && strlen(part1))
	{
		s = part1;
		for (z = 0; z < (int)s.size(); z++)
		{
			if (!isdigit((unsigned char)s.at(z)))
				s[z] = ' ';
		}
		s = _trim(s);
		retval += s;
		retval += ".";
	}
	if (part2 && strlen(part2))
	{
		retval += part2;
		retval += ".";
	}
	retval += dcm_utl_get_datetime();
	return retval;
}
