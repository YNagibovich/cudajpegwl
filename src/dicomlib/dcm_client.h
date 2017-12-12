/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once

// DICOM library specific headers
#include "dcm_version.h"
#include "dcm_data.h"
#include "dcm_constants.h"
#include "dcm_qry.h"

#include "dcm_net.h"
#include <vector>

//bool DCM_CMD_FIND_SCU(char* buffer, dicom_session_context_t* ctx, DCM_QWERY_t* qwery, DCM_QWERY_t* answer, int mode=0);
bool DCM_CMD_ECHO_SCU(char* buffer, dicom_session_context_t* ctx);
//bool DCM_CMD_MOVE_SCU(char* buffer, dicom_session_context_t* ctx, char* destination_AET, DCM_QWERY_t* qwery);
bool DCM_CMD_STORE_SCU(char* buffer, dicom_session_context_t* ctx, image_store_info_t* storage, int mode);

// sent progress notifications to parent window
#define WM_USER_DCIOM_BASE					(WM_APP+200)
#define WM_USER_DCIOM_ECHO_START			WM_USER_DCIOM_BASE+1
#define WM_USER_DCIOM_ECHO_STEP				WM_USER_DCIOM_BASE+2
#define WM_USER_DCIOM_ECHO_STOP				WM_USER_DCIOM_BASE+3

#define WM_USER_DCIOM_SEND_START			WM_USER_DCIOM_BASE+4
#define WM_USER_DCIOM_SEND_STEP				WM_USER_DCIOM_BASE+5
#define WM_USER_DCIOM_SEND_STOP				WM_USER_DCIOM_BASE+6

// High level, SCU
bool DICOMEchoe(HWND parent, std::string sAET, DICOM_PROVIDER_t* info, dcm_logger_cb cb_logger);
bool DICOMSend(HWND parent, std::string sAET, DICOM_PROVIDER_t* info, const char* file_name, dcm_logger_cb cb_logger);

#define ECHO_DEFAULT_TIMEOUT	500 // 0.5 sec