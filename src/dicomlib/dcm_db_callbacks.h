#pragma once

#include "dcm_qry.h"

// DB callbacks

int set_connection_cb(void* ctx);
int close_connection_cb(void* ctx);
char* partner_find_addr_cb(void* user_ctx, char* aet);
int partner_find_port_cb(void* user_ctx, char* aet);
int partner_arc_mode_cb(void* user_ctx, char* aet);
char* create_image_path_cb(void* user_ctx, char* calling_aet, PATIENT_DATA_t* pd, EXAM_DATA_t* ed, SERIES_DATA_t* sd, IMAGE_DATA_t* imd);
int find_image_cb(void* user_ctx, IMAGE_DATA_t* image);
int add_data_cb(void* user_ctx, PATIENT_DATA_t* patient, EXAM_DATA_t* exam, SERIES_DATA_t* series, IMAGE_DATA_t* image);
int qwery_DWL_cb(void* user_ctx, DCM_QWERY_t* qry, DCM_QWERY_t* answer);
int qwery_cb(void* user_ctx, DCM_QWERY_t* qry, DCM_QWERY_t* answer);
int get_storage_info_cb(void* user_ctx, void* data_ctx);
int get_files_cb(void* user_ctx, DCM_QWERY_t* qry, CWordArray* output, int* t_syntax);
int load_DICOM_partners( char* path);
void close_DICOM_DB();