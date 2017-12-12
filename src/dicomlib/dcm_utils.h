/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once
#include "dcm_version.h"
#include "dcm_data.h"
#include "dcm_constants.h"

// externals

// dictionary
extern char *DICOM_UIDs[];
extern DictEntry_t DICOM_DICT[];
extern char* AbstractSyntaxes[];
extern char* TSyntaxes[];

// aux
std::string dcm_utl_get_date();
std::string dcm_utl_get_time();
std::string dcm_utl_get_datetime();

std::string _trim(const std::string& s);

std::wstring s2ws(const std::string& s);

std::string GetTempFile();
std::string GetModalitySOPClassUID(const char* modality);
std::string DCM_Create_GUID(const char* part1, char* part2);

uint32_t sw(uint32_t d);
void sb(uint16_t *d);