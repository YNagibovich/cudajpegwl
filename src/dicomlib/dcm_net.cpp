/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <direct.h>

#include "dcm_version.h"
#include "dcm_net.h"
#include "dcm_utils.h"

//#include "DICOM_DataBase.h"

//#include "FolderUtils.h"
//#include "Iphlpapi.h"
//#include "nic.h"

#define CLEAR_BUFFER memset(buffer,0,MAX_BUFFER_SIZE/4)

static char* Get_own_GUID( dicom_session_context_t* ctx)
{
	//dicom_server_context_t* pctx;

	if(ctx==NULL) 
		return NULL;
	//pctx=( dicom_server_context_t*)ctx->parent;
	//if(pctx==NULL) 
		return ctx->own_guid;
	//return pctx->own_guid;
}

static char* Get_own_AET( dicom_session_context_t* ctx)
{
	//dicom_server_context_t* pctx;

	if(ctx==NULL) 
		return NULL;
	//pctx=(dicom_server_context_t*)ctx->parent;
	//if(pctx==NULL) 
		return &ctx->assotiation_calling_AET[0];
	//return pctx->aet;
}

int Get_own_GUID_len( dicom_session_context_t* ctx)
{
	char* ptr;

	ptr=Get_own_GUID(ctx);
	if(ptr) 
		return strlen(ptr);
	else 
		return 0;
}

int Get_own_AET_len( dicom_session_context_t* ctx)
{
	char* ptr;

	ptr=Get_own_AET(ctx);
	if(ptr) 
		return strlen(ptr);
	else 
		return 0;
}

int DCMSendData( dicom_session_context_t* ctx, char* buf, UINT32 len)
{	
	int ret;
	fd_set	fdWrite  = { 0 };
	TIMEVAL	stTime;
	
	// set timeout
	stTime.tv_sec = ctx->timeout;
	stTime.tv_usec = 0;
	// send data
	if(ctx->timeout)
	{	
		FD_ZERO(&fdWrite);
		FD_SET(ctx->socket, &fdWrite);
		ret=select(NULL, NULL, &fdWrite, NULL, &stTime);
		if(ret==SOCKET_ERROR) return SOCKET_ERROR;
		if(ret==0) return SOCKET_ERROR;
	}
	if(ctx->dumper) ctx->dumper(ctx->session_id, 1, len, buf);
	ret=send(ctx->socket,(char*)buf,len,0);
	if(ret==SOCKET_ERROR) 
	{	
		int err;
		err=WSAGetLastError();
		switch(err)
		{	//case WSAEINTR :
			case WSAENOBUFS :
			case WSAEWOULDBLOCK :	ret=0; break;
			default : ret=SOCKET_ERROR; break;
		}
	}
	return ret;
}

int DCMRecvData( dicom_session_context_t* ctx, char* buf, UINT32 len)
{	
	int ret,err;
	fd_set	fdRead  = { 0 };
	TIMEVAL	stTime;
	
	// set timeout
	stTime.tv_sec = ctx->timeout;
	stTime.tv_usec = 0;
	// send data
	if(ctx->timeout)
	{	
		FD_ZERO(&fdRead);
		FD_SET(ctx->socket,&fdRead);
		ret=select(NULL,&fdRead,NULL,NULL,&stTime);
		if(ret==SOCKET_ERROR) return SOCKET_ERROR;
		if(ret==0) 
		{	
			return SOCKET_ERROR; // check this !!!
		}
	}
    // clear buffer
	memset(buf,0,len);
	ret=recv(ctx->socket,(char*)buf,len,0);
	if(ret==SOCKET_ERROR) 
	{	
		err=WSAGetLastError();
		switch(err)
		{	
			//case WSAEINTR :
			case WSAENOBUFS :
			case WSAEWOULDBLOCK :	ret=0;break;
			default : ret=SOCKET_ERROR;break;
		}
	}
	if(ctx->dumper) 
		ctx->dumper(ctx->session_id, 0, ret, buf);
	if(ret==0) 
		ret=SOCKET_ERROR; // remote socket gracefully closed
	return ret;
}

int DCMCloseConnection( dicom_session_context_t* ctx, int mode)
{	
	int err;

	err=shutdown(ctx->socket, SD_BOTH);
	//if(err==SOCKET_ERROR)	AfxMessageBox("E1");
	err=closesocket(ctx->socket);
	//if(err==SOCKET_ERROR) AfxMessageBox("E2");
	ctx->socket=INVALID_SOCKET;
	return 1; // OK
}

int DCMCheckPartner( dicom_session_context_t* ctx)
{	
	//if(DB.FindProviderAddr(ctx->assotiation_calling_AET)==NULL) return 0;
	//ctx->arc_mode=DB.FindProviderArcMode(ctx->assotiation_calling_AET);
	if(ctx->db_ctx && ctx->db_ctx->partner_find_addr_cb(ctx, ctx->assotiation_calling_AET)==NULL) 
		return 0;
	if(ctx->db_ctx) 
		ctx->arc_mode=ctx->db_ctx->partner_arc_mode_cb(ctx, ctx->assotiation_calling_AET);
	else 
		ctx->arc_mode=0;
	return 1;
}

bool StartAssociation(char* buffer,int size, dicom_session_context_t* ctx)
{	
	PDU_DATA1_t* ptr1;
	PDU_DATA2_t* ptr2;
	ASSOCIATE_RQ_t* aptr;
	char* bptr,*bptr2,*bptr3,*stg;
	UINT16 len1,len,len2,len3;
	int z, acnt, i;
	int aid=0;
	char current_association_name[128];
	
	memset(ctx->association, 0, MAX_ASSOCIATION*sizeof(ASSOCIATION_t));
	// calling aet
	memcpy(&ctx->assotiation_calling_AET[0], (buffer+26), 16);
	for(i=0; i<16; i++)
	{
		if(ctx->assotiation_calling_AET[i]==(char)0x20) 
		{
			ctx->assotiation_calling_AET[i]=0;
		}
	}
	// check partner
	if(!DCMCheckPartner(ctx)) 
	{	
		if(ctx->logger) ctx->logger( DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. Попытка установления ассоциации с незарегистрированным партнёром - %s", 
			ctx->session_id, ctx->assotiation_calling_AET);
		return false;
	}
	bptr=buffer+74;
	ptr1=(PDU_DATA1_t*)bptr;
	//check application context
	if(ptr1->type!=0x10)
	{	
		return false;
	}
	// get length
	len1=ptr1->length;
	sb(&len1);
	z=74+len1+4;
	while(z<size)
	{	bptr+=(len1+4);
		ptr2=(PDU_DATA2_t*)bptr;
		if(ptr2->type!=0x20)
		{	break;
		}
		len1=ptr2->length;
		sb(&len1);
		z+=(len1+4);	
		ctx->association[aid].context_id=ptr2->context_id; // get context_id
		// analyze presentation context
		bptr2=bptr;
		bptr2+=8;
		len=4;
		ptr1=(PDU_DATA1_t*)bptr2;
		len2=ptr1->length;
		sb(&len2);
		if(ptr1->type==0x30)
		{	memset(current_association_name,0,128);
			memcpy(current_association_name,bptr2+4,len2&0x7F);
			memcpy(ctx->association[aid].uid,current_association_name,strlen(current_association_name));
			ctx->association[aid].supported=ASS_NOT_SUPPORTED;
  			while(len<len1)
			{	bptr2+=(len2+4);
				len+=(len2+4);
				ptr1=(PDU_DATA1_t*)(bptr2);
				if(ptr1->type!=0x40) break;
				len2=ptr1->length;
				sb(&len2);
				bptr3=bptr2+4;
				//if(!memcmp("1.2.840.10008.1.2",bptr3,len2))  association[aid].type|=1;//Implicit VR Little Endian
				//if(!memcmp("1.2.840.10008.1.2",bptr3,17))  association[aid].type|=1;//Implicit VR Little Endian
				//else if(!memcmp("1.2.840.10008.1.2.1",bptr3,len2))  association[aid].type|=2; //Explicit VR Little Endian
				//else if(!memcmp("1.2.840.10008.1.2.1.99",bptr3,len2))  association[aid].type|=4; //Deflated Explicit VR Little Endian
				//else if(!memcmp("1.2.840.10008.1.2.2",bptr3,len2))  association[aid].type|=8; //Explicit VR Big Endian
				//else association[aid].type=0;
				for(acnt=0;TSyntaxes[acnt]!=NULL;acnt++)
				{	// found
					//if(!memcmp(bptr3,TSyntaxes[acnt],len2))
					if(!memcmp(bptr3,TSyntaxes[acnt],strlen(TSyntaxes[acnt])))
					{	//memcpy(ctx->association[aid].uid,current_association_name,strlen(current_association_name));
						ctx->association[aid].type=(byte)acnt;
						ctx->association[aid].supported=ASS_SUPPORTED;
		  				if(aid>=MAX_ASSOCIATION) break;
					}
				}
			}
			aid++;
			if(aid>=MAX_ASSOCIATION) break;
		}
		ptr1=(PDU_DATA1_t*)bptr;
	}
	if(!aid) return false;
/*	
	// dump presentation contexts
	FILE* f;
	f=fopen("c:\\ass.txt","wt");
	int i;
	CString ss;
	for(i=0;i<MAX_ASSOCIATION;i++)
	{	ss.Format("%d,%s\n",ctx->association[i].context_id,ctx->association[i].uid);
		fputs(ss.c_str(),f);
	}
	fclose(f);
*/
	// create accept
	aptr=(ASSOCIATE_RQ_t*)buffer;
	aptr->type=02;                    // set type accept
	len=74;
	len1=0;
	bptr=buffer+len;
	// application context
	ptr1=(PDU_DATA1_t*)bptr;
	ptr1->type=0x10;
	ptr1->reserved=0;
	stg="1.2.840.10008.3.1.1.1";
	len3=(UINT16)strlen(stg);
	memcpy(bptr+4,stg,len3);
	bptr+=len3+4;
	len+=len3+4;
	len1+=len3+4;
	sb(&len3);
	ptr1->length=len3;
	// create presentation context
	for(z=0;z<aid;z++)
	{	len1=4;
		ptr2=(PDU_DATA2_t*)bptr;
		ptr2->type=0x21;
		ptr2->reserved1=0;
		ptr2->length=0;
		ptr2->context_id=ctx->association[z].context_id;
		ptr2->reserved2=0;
		//ptr2->reserved3=0; // support syntax
		ptr2->reserved3=ctx->association[z].supported;
		//else ptr2->reserved3=4; // transfer syntax not supported
		ptr2->reserved4=0;
		len+=8;
		bptr+=8; 
		//if(association[z].type&0x01) 
		{	ptr1=(PDU_DATA1_t*)bptr;
			bptr2=bptr+4;
			len3=strlen(TSyntaxes[ctx->association[z].type]);	  
			memcpy(bptr2,TSyntaxes[ctx->association[z].type],len3);//Implicit VR Little Endian
			bptr+=len3+4;
			len+=len3+4;
			len1+=len3+4;
			sb(&len3);
			ptr1->length=len3;
			ptr1->type=0x40;
			ptr1->reserved=0;
		}
		sb(&len1);
		ptr2->length=len1;
	}
	// add user information item
	ptr1=(PDU_DATA1_t*)bptr;
	bptr2=bptr; // save ptr
	ptr1->type=0x50;
	ptr1->reserved=0;
	len+=4;
	len1=0;
	bptr+=4;
	//pdu length
	*bptr++=0x51;
	*bptr++=0x0;
	*bptr++=0x0;
	*bptr++=0x4;
	*bptr++=0x0;
	*bptr++=0x0;
	*bptr++=0x40;
	*bptr++=0x0;
	len+=8;
	len1+=8;
	// own UID
	ptr1=(PDU_DATA1_t*)bptr;
	ptr1->type=0x52;
	ptr1->reserved=0;
	stg=Get_own_GUID(ctx);
	len3=(UINT16)strlen(stg);
	memcpy(bptr+4,stg,len3);
	bptr+=len3+4;
	len+=len3+4;
	len1+=len3+4;
	sb(&len3);
	ptr1->length=len3;
	// own name
	ptr1=(PDU_DATA1_t*)bptr;
	ptr1->type=0x55;
	ptr1->reserved=0;
	stg=Get_own_AET(ctx);
	len3=(UINT16)strlen(stg);
	memcpy(bptr+4,stg,len3);
	len+=len3+4;
	len1+=len3+4;
	sb(&len3);
	ptr1->length=len3;
	// set size
	ptr1=(PDU_DATA1_t*)bptr2;
	sb(&len1);
	ptr1->length=len1;
	aptr->length=len-6;
	sb(&aptr->length);
	// send data
	//z=send(ctx->socket,buffer,len,0);
	z=DCMSendData(ctx,buffer,len);
	if(z==SOCKET_ERROR) return false;
	if(ctx->association[0].type==4) 
	{	
		ctx->assotiation_mode|=ASS_EXPLICIT;
	}
	else ctx->assotiation_mode=0;
	if(ctx->logger) ctx->logger( DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM session #%d. Partner - %s",
		ctx->session_id, ctx->assotiation_calling_AET);
	return true;
}

void DCMAddCommand( char* buffer,int* cmd_length,UINT32 tag,UINT16 data)
{	char* ptr;
	UINT16 len;
	if(!buffer) return;
	ptr=buffer+*cmd_length;
	// copy tag
	memcpy(ptr,&tag,4);
	ptr+=4;
	*cmd_length+=4;
	// copy len
	len=2;
	memcpy(ptr,&len,2);
	ptr+=2;
	*cmd_length+=2;
	// copy reserved
	len=0;
	memcpy(ptr,&len,2);
	ptr+=2;
	*cmd_length+=2;
	// copy data
	memcpy(ptr,&data,2);
	*cmd_length+=2;
}

void DCMAddCommand(char* buffer,int* cmd_length,UINT32 tag,UINT32 data)
{	char* ptr;
	UINT16 len;
	if(!buffer) return;
	ptr=buffer+*cmd_length;
	// copy tag
	memcpy(ptr,&tag,4);
	ptr+=4;
	*cmd_length+=4;
	// copy len
	len=4;
	memcpy(ptr,&len,2);
	ptr+=2;
	*cmd_length+=2;
	// copy reserved
	len=0;
	memcpy(ptr,&len,2);
	ptr+=2;
	*cmd_length+=2;
	// copy data
	memcpy(ptr,&data,4);
	*cmd_length+=4;
}

void DCMAddCommand(char* buffer,int *cmd_length,UINT32 tag, const char* data)
{	char* ptr;
	UINT16 len,val;
	bool correct=false;
	if(!buffer) return;
	ptr=buffer+*cmd_length;
	// copy tag
	memcpy(ptr,&tag,4);
	ptr+=4;
	*cmd_length+=4;
	// copy len
	len=(UINT16)strlen(data);
	if(len&0x01) 
	{	len++;
		correct=true;
	}
	memcpy(ptr,&len,2);
	ptr+=2;
	*cmd_length+=2;
	// copy reserved
	val=0;
	memcpy(ptr,&val,2);
	ptr+=2;
	*cmd_length+=2;
	// copy data
	if(!correct) memcpy(ptr,data,len);
	else
	{	memcpy(ptr,data,len-1);
		*(ptr+len)=0x20;
	}
	*cmd_length+=len;
}

bool DCMSendPDU( dicom_session_context_t* ctx,char* buf,UINT32 len, int mode)
{	int ret;
	UINT32 z;
	// calculate group length
	if(len&0x01) 
		len++;
	// correct length
	if(!mode)
	{	
		z=len-12;
		memcpy(buf+8,&z,2);    
	}
	// send data
	//ret=send(ctx->socket,(char*)buf,len,0);
	ret=DCMSendData(ctx,(char*)buf,len);
	if(ret==SOCKET_ERROR) return false;
	else return true;
}

bool DCMSendPDV( dicom_session_context_t* ctx,UINT32 len,int8_t tag)
{	
	PDU_DATA_TF_EX_t pdv;
	int ret;
	pdv.type=0x04;
	pdv.reserved=0;
	pdv.reserved1=0;
	pdv.reserved2=0;
	pdv.tag=tag;
	if(len&0x01) len++;
	pdv.presentation_ID=ctx->association_PID;
	pdv.length=(UINT16)(len+6);
	pdv.item_length=(UINT16)(len+2);
	sb(&pdv.length);
	sb(&pdv.item_length);
	//ret=send(ctx->socket,(char*)&pdv,12,0);
	ret=DCMSendData(ctx,(char*)&pdv,12);
	if(ret==SOCKET_ERROR) return false;
	else return true;
}

bool DCMEndAssociationRply( dicom_session_context_t* ctx,char* buf) // end association
{	
	int8_t reply[10];
	int ret;
	memset(reply,0,10);
	
	reply[0]=6;
	reply[5]=4;
	//ret=send(ctx->socket,(char*)reply,10,0);
	ret=DCMSendData(ctx,(char*)reply,10);
	if(ret==SOCKET_ERROR) return false;
	else 
	{	
		ctx->status=SESSION_STOPPED;	
		if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM session #%d. Close assoc. Partner - %s", 
			ctx->session_id, ctx->assotiation_calling_AET);
		return true;
	}
}

bool DCMEndAssociationRq( dicom_session_context_t* ctx,char* buf) // end association
{	
	int8_t reply[10];
	int ret;
	
	memset(reply,0,10);
	reply[0]=0x05;
	reply[5]=4;
	//ret=send(ctx->socket,(char*)reply,10,0);
	ret=DCMSendData(ctx,(char*)reply,10);
	if(ret==SOCKET_ERROR) return false;
	else return true;
}

bool DCMEndAssociationGetRply( dicom_session_context_t* ctx,char* buf) // end association
{	
	int8_t reply[10];
	int ret;
	bool retval = false;
	
	memset(reply,0,10);
	//ret=recv(ctx->socket,(char*)reply,10,0);
	ret=DCMRecvData(ctx,(char*)reply,10);
	if(ret==SOCKET_ERROR) retval=false;
	else 
	{	
		if(reply[0]==0x06)
		{	
			ctx->status=SESSION_STOPPED;	
			retval=true;
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM сессия #%d.Ассоциация закрыта. Партнёр - %s", 
				ctx->session_id, ctx->assotiation_calling_AET);
		}
	}
	return retval;
}

int DCM_GET_DATA(char* buffer, dicom_session_context_t* ctx)
{	
	int cmd_len=0;
	int buffer_size;
	int ret,offset,total=0;
	PDU_DATA_TF_EX_t* cmd;
	
	buffer_size=FIRST_PACKET;
	//ret=recv(ctx->socket,buffer,buffer_size,0);
	ret=DCMRecvData(ctx,buffer,buffer_size);
	if(ret==SOCKET_ERROR) return -1;
	cmd=(PDU_DATA_TF_EX_t*)buffer;
	sb(&cmd->length); // correct endianing
	buffer_size=cmd->length-ret+4+2; 
	if(cmd->type==0x06) 
	{	
		ctx->end_ass_rq=1;
		return FIRST_PACKET; // end assoc rq
	}
	else ctx->end_ass_rq=0;
	while(1)
	{	
		if(buffer_size>MAX_BUFFER_SIZE) 
		{	
			ret=SOCKET_ERROR;
			break;
		}
		if(buffer_size<=0) break;
		offset=ret;total=0;
		while(total<buffer_size)
		{	
			//ret=recv(ctx->socket,buffer+total+offset,buffer_size-total,0);
			ret=DCMRecvData(ctx,buffer+total+offset,buffer_size-total);
			if(ret==SOCKET_ERROR) break;  
			total+=ret;
		}
		break;
	}
	if(ret==SOCKET_ERROR) return -1;
	else return total;
}
// tags for IM
static UINT32 IM_EMPTY[]={0x00};			
static UINT32 IM_patient[]={0x00100010,0x00200010,0x00400010,0x00};			
static UINT32 IM_study[]={0x00200008,0x00300008,0x00500008,0x00100010,0x000D0020,0x00100020,0x12060020,0x12080020,0x00};	
static UINT32 IM_P_series[]={0x00600008,0x00110020,0x000E0020,0x00};			
static UINT32 IM_image[]={0x00180008,0x00220008,0x00100018,0x00200018,0x00500018,0x00600018,0x11510018,0x12100018,0x00130020,0x00};			
static UINT32 IM_S_study[]={0x00200008,0x00300008,0x00500008,0x00100010,0x00200010,0x00400010,0x10100010,0x000D0020,0x00100020,0x12060020,0x12080020,0x00};			
static UINT32 IM_S_series[]={0x00600008,0x00110020,0x000E0020,0x00};			

bool DCM_ADD_QRY_TAG(DCM_QWERY_t* qry, uint32_t tag)
{	
	int i;
	DCM_QRY_item_t item;

	for( i=0;i<qry->v_items.size();i++)
	{	
		if(qry->v_items[i].tag==tag) 
			return false; // already exists
	}
	item.tag = tag;
	item.data.clear();
	qry->v_items.push_back(item);
	return true;
}

bool DCM_INIT_QRY_TAGS(DCM_QWERY_t* qry,char* sop_uid)
{	
	int level_idx,i;
	UINT32* tags;
	
	// set level
	if(qry->s_mode=="PATIENT") level_idx=0;
	else if (qry->s_mode == "STUDY") level_idx = 1;
	else if (qry->s_mode == "SERIES") level_idx = 2;
	else if (qry->s_mode == "IMAGE") level_idx = 3;
	else if (qry->s_mode == "INSTANCE") level_idx = 3;
	else level_idx=0;
	// set root
	if((!strcmp(QRY_FIND_PATIENT,sop_uid)) ||(!strcmp(QRY_MOVE_PATIENT,sop_uid)))
	{	
		switch(level_idx)
		{	
		case 0: tags=IM_patient;break;
			case 1: tags=IM_study;break;
			case 2: tags=IM_P_series;break;
			case 3: tags=IM_image;break;
			default : tags=IM_EMPTY;break;
		}
	}
	else if((!strcmp(QRY_FIND_STUDY,sop_uid)) || (!strcmp(QRY_MOVE_STUDY,sop_uid)))
	{	
		switch(level_idx)
		{	
		case 1: tags=IM_S_study;break;
			case 2: tags=IM_S_series;break;
			case 3: tags=IM_image;break;
			default : tags=IM_EMPTY;break;
		}
	}
	else if((!strcmp(QRY_FIND_PAT_STUDY_ONLY,sop_uid)) || (!strcmp(QRY_MOVE_PAT_STUDY_ONLY,sop_uid)))
	{	switch(level_idx)
		{	case 0: tags=IM_patient;break;
			case 1: tags=IM_study;break;
			default : tags=IM_EMPTY;break;
		}
	}
	else return false;
	// infill tags
	//qry->size=0;
	for(i=0;tags[i]!=0;i++) 
	{	
		DCM_ADD_QRY_TAG( qry,tags[i]);
	}
	return true;
}

bool DCM_SET_QRY_TAG(DCM_QWERY_t* qry,UINT32 tag,char* val)
{	int i;
	bool retval=false;
	UINT32 ztag=tag&0xFF;
	if(ztag>0x28) return retval; // check for tag
	//ignore sequence tags
	//if(tag==0x11200008) return retval;
	//if(tag==0x10320008) return retval;
	//if(ztag==0x08) return retval;
	ztag=tag&0xFFFF0000;
	if(ztag==0) 
		return retval;
	for(i=0;i<qry->v_items.size();i++)
	{	
		if(qry->v_items[i].tag==tag)
		{
			qry->v_items[i].data = val;
			retval=true;
		}
	}
	//ignore sequence tags
	ztag=tag&0xFFFF;
	if(ztag==0x08) 
	{	
		ztag=tag&0xFFFF0000;
		switch(ztag)
		{	
			case 0x00180000:
			case 0x00200000:
			case 0x00220000:
			case 0x00300000:			
			case 0x00500000:
			case 0x00600000: break;
			default : return retval;
		}
	}
	if(!retval) // add new tag
	{	
		DCM_QRY_item_t item;

		item.tag=tag;
		item.data=val;
		qry->v_items.push_back(item);
	}
	return retval;
}

void ResetUID( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) 
		return;
	ctx->U_8=0;
	ctx->U_16=0;
	ctx->U_32=0;
}

int8_t GetU8( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) 
		return 0;
	ctx->U_8+=1;
	return ctx->U_8;
}

UINT16 GetU16( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) 
		return 0;
	ctx->U_16+=1;
	return ctx->U_16;
}

UINT16 GetU16Odd( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) 
		return 0;
	ctx->U_16+=1;
	if(ctx->U_16&0x01) 
		return ctx->U_16;
	else 
		ctx->U_16+=1;
	return ctx->U_16;
}

UINT32 GetU32( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) 
		return 0;
	ctx->U_32+=1;
	return ctx->U_32;
}

bool CreateAssociacion(char* buffer, dicom_session_context_t* ctx,char* p_context,int t_syntax)
{	
	ASSOCIATE_RQ_t* arq;
	APPLICATION_CONTEXT_t* actx;
	PRESENTATION_CONTEXT_t* pctx;
	ABSTRACT_SYNTAX_t* asyn;
	TRANSFER_SYNTAX_t* tsyn;
	PDU_DATA1_t* dptr;
	PDU_DATA2_t* aptr;
	PDU_DATA3_t* dptr3;
	char* bptr;
	char* bptr2;
	UINT16 len1,len3;
	int ret,a_len,i,pid;
	int total,offset,err_count;
	int buffer_size;
	int ass_done=0;
	struct sockaddr_in addr;
	HANDLE hndl;
	hndl=CreateEvent(NULL,true,false,NULL);
	// create socket
	// connect
	if((ctx->socket==INVALID_SOCKET) || (ctx->socket==NULL))
	{	
		ctx->socket=socket(AF_INET, SOCK_STREAM,IPPROTO_TCP);
		if(ctx->socket==INVALID_SOCKET)
		{	
			//AfxMessageBox("Невозможно создать сокет отправителя.");
			ctx->err_code = 1;
			return false;
		}
		bool reuse=true;
		setsockopt(ctx->socket,SOL_SOCKET,SO_REUSEADDR,(char*)&reuse,sizeof(bool));
		addr.sin_family=AF_INET;
		addr.sin_port=htons((unsigned short)ctx->remote_port);
		addr.sin_addr.s_addr=get_inet_addr(ctx->remote_addr);
		//try to connect
		for(err_count=0;err_count<RECONNECT_COUNT;)
		{	// add timeout
			
			WaitForSingleObject(hndl,RECONNECT_TIMEOUT);
			if(connect(ctx->socket,(sockaddr*)&addr,sizeof(sockaddr))==SOCKET_ERROR)
			{	
				ret=WSAGetLastError();
				err_count++;
			}
			else 
				break;
		}
		if (err_count == RECONNECT_COUNT)
		{
			ctx->err_code = 2;
			return false;
		}
	}
	CLEAR_BUFFER;
	arq=(ASSOCIATE_RQ_t*)buffer;
	arq->type=0x01;
	arq->resrved1=0;
	arq->version=0x0100;
	arq->reserved2=0;
	memset(arq->CalledAET,0x20,16);
	memset(arq->CallingAET,0x20,16);
	strncpy((char*)arq->CalledAET, ctx->remote_aet, strlen(ctx->remote_aet));
	strncpy((char*)arq->CallingAET, Get_own_AET(ctx), Get_own_AET_len(ctx));
	arq->length=68; // total+6!!!
	// app context
	actx=(APPLICATION_CONTEXT_t*)(buffer+sizeof(ASSOCIATE_RQ_t));
	actx->type=0x10;
	actx->reserved=0;
	actx->length=(UINT16)strlen(DCM_APPLICATION_CONTEXT);
	//if(actx->length&0x01) actx->length+=1;
	strcpy(actx->data,DCM_APPLICATION_CONTEXT);
	arq->length+=4;arq->length+=actx->length;
	for(i=0,pid=1;1;i++,pid+=2)
	{	
		if (AbstractSyntaxes[i] == NULL)
		{
			//ctx->err_code = 3;
			// end of list
			break;
		}
		// presentation context
		pctx=(PRESENTATION_CONTEXT_t*)(buffer+arq->length+6);
		pctx->type=0x20;
		pctx->context_id=(int8_t)pid;
		arq->length+=sizeof(PRESENTATION_CONTEXT_t);
		// abstract syntax
		asyn=(ABSTRACT_SYNTAX_t*)(buffer+arq->length+6);
		asyn->type=0x30;
		if(p_context)
		{	
			asyn->length=(UINT16)strlen(p_context);
			strcpy(asyn->data,p_context);
		}
		else
		{	
			asyn->length=(UINT16)strlen(AbstractSyntaxes[i]);
			strcpy(asyn->data,AbstractSyntaxes[i]);
		}
		arq->length+=4;arq->length+=asyn->length;
		// transfer syntax
		tsyn=(TRANSFER_SYNTAX_t*)(buffer+arq->length+6);
		tsyn->type=0x40;
		//tsyn->length=(UINT16)strlen(DCM_TRANSFER_SYNTAX);
		//strcpy(tsyn->data,DCM_TRANSFER_SYNTAX);
		tsyn->length=(UINT16)strlen(TSyntaxes[t_syntax]);
		strcpy(tsyn->data,TSyntaxes[t_syntax]);
		//if(tsyn->length&0x01) tsyn->length+=1;
		arq->length+=4;arq->length+=tsyn->length;
		pctx->length=4+asyn->length+4+tsyn->length+4;
		sb(&pctx->length);
		sb(&asyn->length);
		sb(&tsyn->length);
		if(p_context) 
			break;
	}
	// user items;
	bptr=buffer+arq->length+6;
	dptr=(PDU_DATA1_t*)bptr;
	bptr2=bptr; // save ptr
	dptr->type=0x50;
	dptr->reserved=0;
	arq->length+=4;
	len1=0;
	bptr+=4;
	//pdu length // use 16384 , may be larger
	*bptr++=0x51;
	*bptr++=0x0;
	*bptr++=0x0;
	*bptr++=0x4;
	*bptr++=0x0;
	*bptr++=0x0;
	*bptr++=0x40;
	*bptr++=0x0;
	arq->length+=8;
	len1+=8;
	// own UID
	dptr=(PDU_DATA1_t*)bptr;
	dptr->type=0x52;
	dptr->reserved=0;
	len3=Get_own_GUID_len(ctx);
	memcpy(bptr+4, Get_own_GUID(ctx), len3);
	if(len3&0x01) 
		len3++;
	bptr+=len3+4;
	arq->length+=len3+4;
	len1+=len3+4;
	sb(&len3);
	dptr->length=len3;
	// own name
	dptr=(PDU_DATA1_t*)bptr;
	dptr->type=0x55;
	dptr->reserved=0;
	len3=(UINT16)Get_own_AET_len(ctx);
	memcpy(bptr+4, Get_own_AET(ctx), len3);
	if(len3&0x01) len3++;
	arq->length+=len3+4;
	len1+=len3+4;
	sb(&len3);
	dptr->length=len3;
	// set size
	dptr=(PDU_DATA1_t*)bptr2;
	sb(&len1);
	dptr->length=len1;
	// correct lengths
	a_len=arq->length+6;
	sb(&arq->length);
	sb(&actx->length);
	// send data
	//ret=send(ctx->socket,buffer,a_len,0);
	ret=DCMSendData(ctx,buffer,a_len);
	if(ret==SOCKET_ERROR) 
	{	
		ctx->err_code = 4;
		return false;
	}
	// receive ack
	CLEAR_BUFFER;
	while(1)
	{	
		buffer_size=FIRST_PACKET;
		//ret=recv(ctx->socket,buffer,buffer_size,0);
		ret=DCMRecvData(ctx,buffer,buffer_size);
		if(ret==SOCKET_ERROR) 
		{	
			ret=WSAGetLastError();
			ctx->err_code = 5;
			break;
		}
		dptr3=(PDU_DATA3_t*)buffer;
		sb(&dptr3->length);
		buffer_size=dptr3->length-ret+4+2; //??
		if(buffer_size>MAX_BUFFER_SIZE) 
		{	
			break;
		}
		if(buffer_size<=0) 
		{	
			break;
		}
		offset=ret;total=0;
		while(total<buffer_size)
		{	
			//ret=recv(ctx->socket,buffer+total+offset,buffer_size-total,0);
			ret=DCMRecvData(ctx,buffer+total+offset,buffer_size-total);
			if(ret==SOCKET_ERROR) 
			{	
				ctx->err_code = 6;
				break;
			}
			total+=ret;
		}
		if(ret==SOCKET_ERROR) 
		{	
			ctx->err_code = 7;
			break;
		}
		if(dptr3->type==0x02) // Associate AC
		{	
			// get ass pid
			len1=74;
			//skip application context item
			aptr=(PDU_DATA2_t*)(buffer+len1);
			sb(&aptr->length);
			len1+=(4+aptr->length);
			//while(len1<=total)
			while(len1<buffer_size)
			{	
				aptr=(PDU_DATA2_t*)(buffer+len1);
				if(aptr->type==0x21)
				{	
					// 0 - acceptance
					// 1 - user - rejection
					// 2 - no - reason(provider rejection)
					// 3 - abstract - syntax - not - supported(provider rejection)
					// 4 - transfer - syntaxes - not - supported(provider rejection)
					if(aptr->reserved3==0) // accepted
					{	
						ctx->association_PID=aptr->context_id;
						ass_done=1;
						break;
					}
				}
				sb(&aptr->length);
				len1+=(4+aptr->length);
			}
			if(ass_done) 
				break;
		}
		else if(dptr3->type==0x07) // association abort
		{	
			ass_done=0;
			ctx->err_code = 8;
			break;
		}
		else if(dptr3->type==0x03) // association reject
		{	
			ass_done=0;
			ctx->err_code = 9;
			break;
		}
		if(ass_done==0)
			break;
	} 
	if(ass_done) 
		return true;
	else 
		return false;
}

int GetSOPId(char* sop)
{	
	int i;
	int retval=7; // CT image //TBD
	
	if(sop==0) 
		return retval;
	for(i=0; AbstractSyntaxes[i]!=NULL; i++)
	{	
		if(!strncmp(sop, AbstractSyntaxes[i], strlen(sop)))
		{	
			retval=i;
			break;
		}
	}
	return retval;
}

unsigned long get_inet_addr(char* host_name)
{
	hostent* remoteHost;
	unsigned long retval;

	if (strlen(host_name) == 0)
		return 0;
	if (isalpha((unsigned char)host_name[0]))
	{
		remoteHost = gethostbyname(host_name);
		//getaddrinfo()
		if (remoteHost == NULL) 
			return 0;
		retval = *(unsigned long*)remoteHost->h_addr;
	}
	else
	{
		retval = inet_addr(host_name);
	}
	return retval;
}
