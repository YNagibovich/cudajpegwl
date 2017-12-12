/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#include "dcm_version.h"
#include "dcm_srv.h"
#include "dcm_wsi.h"
#include "dcm_file.h"
#include "dcm_utils.h"
#include <string.h>
#include <stdio.h>
#include <direct.h>

//#include "DICOM_DataBase.h"

//#include "FolderUtils.h"
//#include "Iphlpapi.h"
//#include "nic.h"

// internals
DWORD WINAPI DCMServerProc(PVOID pParameter);
DWORD WINAPI DCMSessionProc(PVOID pParameter);
int DCM_GET_DATA(char* buffer, dicom_session_context_t* ctx);
int8_t GetU8( dicom_session_context_t* ctx);
UINT16 GetU16( dicom_session_context_t* ctx);
UINT16 GetU16Odd( dicom_session_context_t* ctx);
UINT32 GetU32( dicom_session_context_t* ctx);


#define CLEAR_BUFFER memset(buffer,0,MAX_BUFFER_SIZE/4)

static char* Get_own_GUID( dicom_session_context_t* ctx)
{
	dicom_server_context_t* pctx;

	if(ctx==NULL) 
		return NULL;
	pctx=( dicom_server_context_t*)ctx->parent;
	if(pctx==NULL) 
		return ctx->own_guid;
	return pctx->own_guid;
}

static char* Get_own_AET( dicom_session_context_t* ctx)
{
	dicom_server_context_t* pctx;

	if(ctx==NULL) 
		return NULL;
	pctx=(dicom_server_context_t*)ctx->parent;
	if(pctx==NULL) 
		return &ctx->assotiation_calling_AET[0];
	return pctx->aet;
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

BOOL StartAssociation(char* buffer,int size, dicom_session_context_t* ctx)
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
		return FALSE;
	}
	bptr=buffer+74;
	ptr1=(PDU_DATA1_t*)bptr;
	//check application context
	if(ptr1->type!=0x10)
	{	
		return FALSE;
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
	if(!aid) return FALSE;
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
	if(z==SOCKET_ERROR) return FALSE;
	if(ctx->association[0].type==4) 
	{	
		ctx->assotiation_mode|=ASS_EXPLICIT;
	}
	else ctx->assotiation_mode=0;
	if(ctx->logger) ctx->logger( DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM session #%d. Partner - %s",
		ctx->session_id, ctx->assotiation_calling_AET);
	return TRUE;
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
	BOOL correct=FALSE;
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
		correct=TRUE;
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

BOOL DCMSendPDU( dicom_session_context_t* ctx,char* buf,UINT32 len,int mode=0)
{	int ret;
	UINT32 z;
	// calculate group length
	if(len&0x01) len++;
	// correct length
	if(!mode)
	{	z=len-12;
		memcpy(buf+8,&z,2);    
	}
	// send data
	//ret=send(ctx->socket,(char*)buf,len,0);
	ret=DCMSendData(ctx,(char*)buf,len);
	if(ret==SOCKET_ERROR) return FALSE;
	else return TRUE;
}

BOOL DCMSendPDV( dicom_session_context_t* ctx,UINT32 len,int8_t tag)
{	PDU_DATA_TF_EX_t pdv;
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
	if(ret==SOCKET_ERROR) return FALSE;
	else return TRUE;
}

BOOL DCMEndAssociationRply( dicom_session_context_t* ctx,char* buf) // end association
{	
	int8_t reply[10];
	int ret;
	memset(reply,0,10);
	
	reply[0]=6;
	reply[5]=4;
	//ret=send(ctx->socket,(char*)reply,10,0);
	ret=DCMSendData(ctx,(char*)reply,10);
	if(ret==SOCKET_ERROR) return FALSE;
	else 
	{	
		ctx->status=SESSION_STOPPED;	
		if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM session #%d. Close assoc. Partner - %s", 
			ctx->session_id, ctx->assotiation_calling_AET);
		return TRUE;
	}
}

BOOL DCMEndAssociationRq( dicom_session_context_t* ctx,char* buf) // end association
{	
	int8_t reply[10];
	int ret;
	
	memset(reply,0,10);
	reply[0]=0x05;
	reply[5]=4;
	//ret=send(ctx->socket,(char*)reply,10,0);
	ret=DCMSendData(ctx,(char*)reply,10);
	if(ret==SOCKET_ERROR) return FALSE;
	else return TRUE;
}

BOOL DCMEndAssociationGetRply( dicom_session_context_t* ctx,char* buf) // end association
{	
	int8_t reply[10];
	int ret;
	BOOL retval = FALSE;
	
	memset(reply,0,10);
	//ret=recv(ctx->socket,(char*)reply,10,0);
	ret=DCMRecvData(ctx,(char*)reply,10);
	if(ret==SOCKET_ERROR) retval=FALSE;
	else 
	{	
		if(reply[0]==0x06)
		{	
			ctx->status=SESSION_STOPPED;	
			retval=TRUE;
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM сессия #%d.Ассоциация закрыта. Партнёр - %s", 
				ctx->session_id, ctx->assotiation_calling_AET);
		}
	}
	return retval;
}

BOOL DCM_CMD_ECHO_SCP(char* buffer, dicom_session_context_t* ctx) //echo reply
{	
	int cmd_len=0;
	int len,l,z;
	int echo_message_id = 0;
	char* bptr=buffer;
	DATA_SET_CMD_t* ptr;
	
	len=512;
	for(l=0,bptr=buffer+12;l<len;)
	{	
		ptr=(DATA_SET_CMD_t*)bptr;
		if(ctx->assotiation_mode&ASS_EXPLICIT) 
			z=ptr->reserved;  
		else 
			z=ptr->val_size;
		if(ptr->tag==0x00000000) 
			len=ptr->data;
		//else if(ptr->tag==0x00020000) memcpy(store_sop_uid,&ptr->data,z);
		else if(ptr->tag==0x01100000) 
			echo_message_id=ptr->data;
		bptr+=z+8;
		l+=z+8;
	}
	CLEAR_BUFFER;
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00020000,ECHO_SOP);
	DCMAddCommand(buffer,&cmd_len,0x01000000,(UINT16)0x8030);
	DCMAddCommand(buffer,&cmd_len,0x01200000,(UINT16)echo_message_id); 
	DCMAddCommand(buffer,&cmd_len,0x08000000,(UINT16)0x0101);
	DCMAddCommand(buffer,&cmd_len,0x09000000,(UINT16)0x0);
	// send it
	if( !DCMSendPDV(ctx,cmd_len,3)) 
		return FALSE;
	if( !DCMSendPDU(ctx,buffer,cmd_len)) 
		return FALSE;
#ifdef DICOM_SERVER
	if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM сессия #%d. ECHO SCP - OK from %s", ctx->session_id, ctx->assotiation_calling_AET);
#endif
	return TRUE;
}

BOOL DCM_CMD_STORE_SCP(char* buffer, dicom_session_context_t* ctx) //store 
{	
	int l,ll,len;
	int z;
	int buffer_size;
	int ret;
	int total=0,offset=0;
	int header=1;
	std::string path,file_path;
	DATA_SET_CMD_t* ptr;
	PDU_DATA_TF_EX_t* dptr;
	PATIENT_DATA_t pd;
	EXAM_DATA_t ed;
	SERIES_DATA_t sd;
	IMAGE_DATA_t imd;
	char* bptr=buffer;
	FILE* file=NULL;
	char store_sop_uid[64];
	char store_instance_uid[64];
	char stg[128];
	UINT16 store_message_id=0;
	std::string rep;
	
	memset(store_sop_uid,0,64);
	memset(store_instance_uid,0,64);
	// clear data
	clear_data(&pd);
	clear_data(&ed);
	clear_data(&sd);
	clear_data(&imd);
//#ifndef DICOM_SERVER	
//	CONFIG.pending_context=ctx;
//#endif
	for(l=0,bptr=buffer+12;l<1024;)
	{	
		ptr=(DATA_SET_CMD_t*)bptr;
		z=ptr->val_size;
		if(ptr->tag==0x00020000) memcpy(store_sop_uid,&ptr->data,z);
		else if(ptr->tag==0x01100000) store_message_id=ptr->data;
		else if(ptr->tag==0x10000000) 
		{	
			memcpy(store_instance_uid,&ptr->data,z);
		}
		else if(ptr->tag==0x10300000) // MOVE AET
		{	
			ll=0;
		}
		else if(ptr->tag==0x10310000) // MOVE message ID
		{	
			ll=0;
		}
		bptr+=ptr->val_size+8;
		l+=ptr->val_size+8;
	}
	// get data
	l=0;z=0;
	while(!l)
	{	
		buffer_size=FIRST_PACKET;
		ret=DCMRecvData(ctx,buffer,buffer_size);
		if(ret==0) 
		{	
			continue;
		}
		if(ret==SOCKET_ERROR) 
		{	
			break;
		}
	    dptr=(PDU_DATA_TF_EX_t*)buffer;
		sb(&dptr->length); // correct endianing
		buffer_size=dptr->length-ret+4+2; 
		if(buffer_size>MAX_BUFFER_SIZE) 
		{	
			break;
		}
		if(buffer_size<=0) 
		{	
			continue;
		}
		total=0;offset=ret;
		while(total<buffer_size)
		{	
			ret=DCMRecvData(ctx,buffer+total+offset,buffer_size-total);
			total+=ret;
			if(ret==SOCKET_ERROR) break;
		}
		if(ret==SOCKET_ERROR) 
		{	
			break;
		}
		if(dptr->tag&0x02) // last data packet
		{	
			l=1;
		}
		if(header)
		{	
			header=0;
			pd.PatientID.clear();
			pd.name.clear();
			ed.name.clear();
			ed.ExamID.clear();
			sd.name.clear();
			sd.SeriesID.clear();
			imd.ImageID.clear();
			// analyze header
			for(ll=0,bptr=buffer+12;ll<buffer_size;)
			{	
				ptr=(DATA_SET_CMD_t*)bptr;
				if(ctx->assotiation_mode&ASS_EXPLICIT) 
				{	
					if(ptr->val_size==0x5153) 
					{	
						len=ptr->data+4;
					}
					else len=ptr->reserved;
				}
				else len=ptr->val_size;
				if(len>127) z=127;
				else z=len;
				memset(stg,0,128);
				memcpy(stg,&ptr->data,z);
				switch(ptr->tag)
				{	
					// patient
					case 0x00100010 : pd.name=stg; break;
					case 0x00200010 : pd.PatientID=stg; break;
					case 0x00300010 : pd.BirthDate=stg; break;
					case 0x00400010 : pd.Sex=stg; break;
					case 0x10100010 : pd.Age=stg; break;
					// exam-study
					case 0x00100020 : ed.name=stg; break;
					case 0x000D0020 : ed.ExamID=stg; break;
					case 0x00200008 : ed.StudyDate=stg; break; // Study Date
					case 0x00300008 : ed.StudyTime=stg; break; // Study Time
					case 0x10300008 : ed.Description=stg; break; // Study Time
					case 0x00500008 : ed.AccessionNumber=stg; break; // Accession Number
					case 0x00600008 : ed.Modality=stg; break; // Modality
					case 0x00610008 : ed.Modality=stg; break; // Modality
					case 0x00150018 : ed.BodyPart=stg; sd.BodyPart=stg; break;
					// series
					case 0x00120020 : sd.name=stg; break;
					case 0x00110020 : sd.Number=stg; break;
					case 0x000E0020 : sd.SeriesID=stg; break;
					case 0x103E0008 : sd.Description=stg; break;
					// image
					case 0x00180008 : imd.ImageID=stg; break;
					case 0x00160008 : imd.SopUID=stg; break;
     				default : break;
				}
				//bptr+=ptr->val_size+8;
				//ll+=ptr->val_size+8;
				bptr+=(len+8);
				ll+=(len+8);
			}
			// handle DB
			//file_path=DB.CreateImagePath(ctx->assotiation_calling_AET, &pd, &ed, &sd, &imd);
			if(ctx->db_ctx) file_path=ctx->db_ctx->create_image_path_cb(ctx, ctx->assotiation_calling_AET, &pd, &ed, &sd, &imd);
			else file_path.clear();
			file=fopen( file_path.c_str(), "wb");
			if(file==NULL)
			{	
				if(ctx->logger) ctx->logger( DCM_LOG_DATA_EVENT, DCM_LOG_ERROR, "DICOM session #%d. Cannot create file : %s", 
					ctx->session_id, file_path.c_str());
			}
		}
		if(file) 
		{	
			fwrite(buffer+12,1,dptr->length-6,file);
		}
	}
	if(file) 
	{	
		fclose(file);
#ifdef DICOM_SERVER
		if(ctx->logger) ctx->logger(DCM_LOG_DATA_EVENT, DCM_LOG_OK, "DICOM сессия #%d. Получен файл : %s", ctx->session_id, file_path);
		// set arc attribute
#endif
		// add to db if no such image there
		if(ctx->db_ctx && ctx->db_ctx->find_image_cb(ctx, &imd)<0) 
		{ 
#ifndef DICOM_SERVER
			//sd.tag=VIEWER_SHOW_SERIE;
			sd.tag=1;
#endif
			//if(store_in_arc>0)
			//{	
			//	pd.state|=STATE_ARC_STORE;
			//	ed.state|=STATE_ARC_STORE;
			//	sd.state|=STATE_ARC_STORE;
			//	imd.state|=STATE_ARC_STORE;
			//}
			//DB.Add(&pd,&ed,&sd,&imd);
			ctx->db_ctx->add_data_cb(ctx, &pd, &ed, &sd, &imd);
		}
	}
	else
	{	
		if(ctx->logger) ctx->logger(DCM_LOG_DATA_EVENT, DCM_LOG_ERROR, "DICOM session #%d. Cannot create file : %s", ctx->session_id, file_path);
	}
	// send ack
	CLEAR_BUFFER;
	l=0;
	DCMAddCommand(buffer,&l,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&l,0x00020000,(char*)store_sop_uid);
	DCMAddCommand(buffer,&l,0x01000000,(UINT16)0x8001);
	DCMAddCommand(buffer,&l,0x01200000,store_message_id); // message id
	DCMAddCommand(buffer,&l,0x08000000,(UINT16)0x0101);
	DCMAddCommand(buffer,&l,0x09000000,(UINT16)0x0); // STATUS
	DCMAddCommand(buffer,&l,0x10000000,(char*)store_instance_uid);
	if(!DCMSendPDV(ctx,l,3)) 
	{	
		return FALSE;
	}
	if(!DCMSendPDU(ctx,buffer,l)) 
	{	
		return FALSE;
	}
	return TRUE;
}

void ClearDCMQwery(DCM_QWERY_t* qry)
{	
	int i;

	qry->size=0;
	//qry->mode.clear();
	for(i=0;i<MAX_QRY;i++) 
	{	
		qry->element[i].data.clear();
		qry->element[i].tag=0;
	}
	qry->patient.clear();
	qry->patient_id.clear();
	qry->study_id.clear();
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

BOOL DCM_CMD_FIND_SCU(char* buffer,  dicom_session_context_t* ctx,DCM_QWERY_t* qwery,DCM_QWERY_t* answer,int* answer_size,int mode)
{	
	int cmd_len=0,l,cnt;
	int buffer_size;
	int last_data,z,ans_cnt;
	DATA_SET_CMD_t* ptr;
	char* bptr;
	PDU_DATA_TF_EX_t* pdv;
	//DCM_QWERY_t answer[MAX_QRY_RESULT];
	char stg[128];
	CLEAR_BUFFER;
	// qry rq
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	if(!mode) DCMAddCommand(buffer,&cmd_len,0x00020000,QRY_FIND_STUDY);
	else DCMAddCommand(buffer,&cmd_len,0x00020000,QRY_FIND_WL);
	DCMAddCommand(buffer,&cmd_len,0x01000000,(UINT16)0x0020); // find rq
	DCMAddCommand(buffer,&cmd_len,0x01100000, GetU16Odd(ctx)); // message id
	DCMAddCommand(buffer,&cmd_len,0x07000000,(UINT16)0x0); // proirity
	DCMAddCommand(buffer,&cmd_len,0x08000000,(UINT16)0x0102);
	DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	DCMAddCommand(buffer,&cmd_len,0x00520008,qwery->mode.c_str());// Query Level
	for(cnt=0;cnt<qwery->size;cnt++)
	{	DCMAddCommand(buffer,&cmd_len,qwery->element[cnt].tag,qwery->element[cnt].data.c_str());
	}
	if(!DCMSendPDV(ctx,cmd_len,2)) // data
	{	if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE; // data_set
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// step 2
	cmd_len=0;
	// qery data
	CLEAR_BUFFER;
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	DCMAddCommand(buffer,&cmd_len,0x00520008,qwery->mode.c_str());// Query Level
	for(cnt=0;cnt<qwery->size;cnt++)
	{	DCMAddCommand(buffer,&cmd_len,qwery->element[cnt].tag,qwery->element[cnt].data.c_str());
	}
	if(!DCMSendPDV(ctx,cmd_len,2)) // data
	{	if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE; // data_set
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// receive ack
	ans_cnt=0;
	while(1)
	{	buffer_size=DCM_GET_DATA(buffer,ctx);

/*
	{	FILE* z;
		z=fopen("c:\\zzz","a+b");
		fwrite(buffer,buffer_size,1,z);
		fclose(z);

	}
*/
		if(buffer_size<0) break;
		pdv=(PDU_DATA_TF_EX_t*)buffer;
		// reply analyzer
		for(l=0,cnt=0,last_data=0,bptr=buffer+12;l<buffer_size;)
		{	ptr=(DATA_SET_CMD_t*)bptr;
			if(ctx->assotiation_mode&ASS_EXPLICIT) z=ptr->reserved;  
			else z=ptr->val_size;
			if(pdv->tag&0x01) //cmd
			{	if(ptr->tag==0x08000000) 
				{	if(ptr->data==0x0101)
					{	last_data=1;
						break;
					}
				}
				else if(ptr->tag==0x01000000)
				{	if(ptr->data!=0x8020) // wrong reply
					{	last_data=1;
						break;
				    }
				}
			}
			else
			{	if(ans_cnt<MAX_QRY_RESULT)
				{	memset(stg,0,128);
					memcpy(stg,&ptr->data,z&0x7F);
					(answer+ans_cnt)->element[cnt].tag=ptr->tag;
					(answer+ans_cnt)->element[cnt].data=stg;
					(answer+ans_cnt)->size++;
					if(cnt<(MAX_QRY-1)) cnt++;
				}
			}
			bptr+=z+8;
			l+=z+8;
		}
		if(!(pdv->tag&0x01)) ans_cnt++;
		if((pdv->tag==0x03) && last_data) break;
	}
	if(answer_size) *answer_size=ans_cnt;
	if(ctx->event_exit) SetEvent(ctx->event_exit);
	return TRUE;
}

BOOL DCM_CMD_ECHO_SCU(char* buffer, dicom_session_context_t* ctx)
{	int cmd_len=0,l,cnt;
	int buffer_size;
	int last_data,z;
	BOOL retval=FALSE;
	DATA_SET_CMD_t* ptr;
	char* bptr;
	PDU_DATA_TF_EX_t* pdv;
	CLEAR_BUFFER;
	// echo rq
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00020000,ECHO_SOP);
	DCMAddCommand(buffer,&cmd_len,0x01000000,(UINT16)0x0030); // echo rq
	DCMAddCommand(buffer,&cmd_len,0x01100000,(UINT16)GetU16Odd(ctx)); // message id
	DCMAddCommand(buffer,&cmd_len,0x07000000,(UINT16)0x0); // proirity
	DCMAddCommand(buffer,&cmd_len,0x08000000,(UINT16)0x0101); // no data
	//DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	if(!DCMSendPDV(ctx,cmd_len,3)) 
	{	if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// receive ack
	while(1)
	{	buffer_size=DCM_GET_DATA(buffer,ctx);
		if(buffer_size<0) break;
		pdv=(PDU_DATA_TF_EX_t*)buffer;
		// reply analyzer
		for(l=0,cnt=0,last_data=0,bptr=buffer+12;l<buffer_size;)
		{	ptr=(DATA_SET_CMD_t*)bptr;
			if(ctx->assotiation_mode&ASS_EXPLICIT) z=ptr->reserved;  
			else z=ptr->val_size;
			if(pdv->tag&0x01) //cmd
			{	if(ptr->tag==0x08000000) 
				{	if(ptr->data==0x0101)
					{	last_data=1;
						break;
					}
				}
				else if(ptr->tag==0x01000000)
				{	if(ptr->data!=0x8030) retval=FALSE; 
					else retval=TRUE; // reply is ok
					last_data=1;
					break;
				}
			}
			bptr+=z+8;
			l+=z+8;
		}
		if((pdv->tag==0x03) && last_data) break;
	}
	if(ctx->event_exit) SetEvent(ctx->event_exit);
	return retval;
}

// tags for IM
static UINT32 IM_EMPTY[]={0x00};			
static UINT32 IM_patient[]={0x00100010,0x00200010,0x00400010,0x00};			
static UINT32 IM_study[]={0x00200008,0x00300008,0x00500008,0x00100010,0x000D0020,0x00100020,0x12060020,0x12080020,0x00};	
static UINT32 IM_P_series[]={0x00600008,0x00110020,0x000E0020,0x00};			
static UINT32 IM_image[]={0x00180008,0x00220008,0x00100018,0x00200018,0x00500018,0x00600018,0x11510018,0x12100018,0x00130020,0x00};			
static UINT32 IM_S_study[]={0x00200008,0x00300008,0x00500008,0x00100010,0x00200010,0x00400010,0x10100010,0x000D0020,0x00100020,0x12060020,0x12080020,0x00};			
static UINT32 IM_S_series[]={0x00600008,0x00110020,0x000E0020,0x00};			

BOOL DCM_ADD_QRY_TAG(DCM_QWERY_t* qry,UINT32 tag)
{	
	int i;
	BOOL retval=FALSE;

	for( i=0;i<qry->size;i++)
	{	
		if(qry->element[i].tag==tag) return FALSE; // already exists
	}
	if(qry->size<(MAX_QRY-1))
	{	
		qry->element[qry->size].tag=tag;
		qry->element[qry->size].data.clear();
		retval=TRUE;
	}
	return retval;
}

BOOL DCM_INIT_QRY_TAGS(DCM_QWERY_t* qry,char* sop_uid)
{	
	int level_idx,i;
	UINT32* tags;
	
	// set level
	if(qry->mode=="PATIENT") level_idx=0;
	else if(qry->mode=="STUDY") level_idx=1;
	else if(qry->mode=="SERIES") level_idx=2;
	else if(qry->mode=="IMAGE") level_idx=3;
	else if(qry->mode=="INSTANCE") level_idx=3;
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
	else return FALSE;
	// infill tags
	qry->size=0;
	for(i=0;tags[i]!=0;i++) 
	{	if(DCM_ADD_QRY_TAG(qry,tags[i])) qry->size++;
	}
	return TRUE;
}

BOOL DCM_SET_QRY_TAG(DCM_QWERY_t* qry,UINT32 tag,char* val)
{	int i;
	BOOL retval=FALSE;
	UINT32 ztag=tag&0xFF;
	if(ztag>0x28) return retval; // check for tag
	//ignore sequence tags
	//if(tag==0x11200008) return retval;
	//if(tag==0x10320008) return retval;
	//if(ztag==0x08) return retval;
	ztag=tag&0xFFFF0000;
	if(ztag==0) return retval;
	for(i=0;i<qry->size;i++)
	{	if(qry->element[i].tag==tag)
		{	qry->element[i].data=val;
			retval=TRUE;
		}
	}
	//ignore sequence tags
	ztag=tag&0xFFFF;
	if(ztag==0x08) 
	{	ztag=tag&0xFFFF0000;
		switch(ztag)
		{	case 0x00180000:
			case 0x00200000:
			case 0x00220000:
			case 0x00300000:			
			case 0x00500000:
			case 0x00600000: break;
			default : return retval;
		}
	}
	if(!retval) // add new tag
	{	if(qry->size<(MAX_QRY-1))
		{	qry->element[qry->size].tag=tag;
			qry->element[qry->size].data=val;
			qry->size++;
		}
	}
	return retval;
}

BOOL DCM_CMD_FIND_SCP(char* buffer, dicom_session_context_t* ctx)
{	
	int l,z,i,j,ret,total,offset,len;
	int buffer_size;
	DCM_QWERY_t qry;
	DCM_QWERY_t answer[MAX_QRY_RESULT];
	DATA_SET_CMD_t* ptr;
	PDU_DATA_TF_EX_t* dptr;
	char* bptr=buffer;
	char store_sop_uid[64];
	char store_instance_uid[64];
	char stg[128];
	BOOL wl_qry=FALSE;
	UINT16 store_message_id=0;
	memset(store_sop_uid,0,64);
	memset(store_instance_uid,0,64);
	len=512;
	for(l=0,bptr=buffer+12;l<len;)
	{	ptr=(DATA_SET_CMD_t*)bptr;
		if(ctx->assotiation_mode&ASS_EXPLICIT) z=ptr->reserved;  
		else z=ptr->val_size;
		if(ptr->tag==0x00000000) len=ptr->data;
		else if(ptr->tag==0x00020000) memcpy(store_sop_uid,&ptr->data,z);
		else if(ptr->tag==0x01100000) store_message_id=ptr->data;
		else if(ptr->tag==0x10000000) memcpy(store_instance_uid,&ptr->data,z);
		bptr+=z+8;
		l+=z+8;
	}
	// get query
	l=0;
	ClearDCMQwery(&qry);
	while(!l)
	{	buffer_size=FIRST_PACKET;
		//ret=recv(ctx->socket,buffer,buffer_size,0);
		ret=DCMRecvData(ctx,buffer,buffer_size);
		if(ret==SOCKET_ERROR) break;
		dptr=(PDU_DATA_TF_EX_t*)buffer;
		sb(&dptr->length); // correct endianing
		buffer_size=dptr->length-ret+4+2; 
		if(buffer_size>MAX_BUFFER_SIZE) break;
		if(buffer_size<=0) break;
		offset=ret;total=0;
		while(total<buffer_size)
		{	//ret=recv(ctx->socket,buffer+total+offset,buffer_size-total,0);
			ret=DCMRecvData(ctx,buffer+total+offset,buffer_size-total);
			total+=ret;
			if(ret==SOCKET_ERROR) break;
		}
		//ret=recv(ctx->socket,buffer+ret,buffer_size,0);
		if(ret==SOCKET_ERROR) break;
		l=1;
	}
	// analyze query
	for(l=0, bptr=buffer+12; l<buffer_size;)
	{	ptr=(DATA_SET_CMD_t*)bptr;
		if(ctx->assotiation_mode&ASS_EXPLICIT) z=ptr->reserved;  
		else z=ptr->val_size;
		if(z==0xFFFF) z=0;
		if(qry.size<MAX_QRY)	
		{	if(z>127) z=127;
			memset(stg, 0, 128);
			memcpy(stg, &ptr->data, z);
			switch(ptr->tag)
			{	case 0x00010000 :	break;
				case 0x00001000 :	break;
				case 0x00008000 :	break;
				case 0x00520008 :	qry.mode=stg; DCM_INIT_QRY_TAGS(&qry, store_sop_uid); break; 
				default :	DCM_SET_QRY_TAG(&qry, ptr->tag, stg);break;
			}
		}
		bptr+=z+8;
		l+=z+8;
	}
	// execute qry
	z=0;
#ifdef DICOM_SERVER
	// DICOM WORKLIST support
	if(!strcmp(QRY_FIND_WL, store_sop_uid))
	{	// WL qwery
#ifndef DEMO

#ifdef ENABLE_FIND_WL_SCP
		wl_qry=TRUE;
		//z=DB.QweryDWL(&qry, &answer[0]);
		if(ctx->db_ctx) z=ctx->db_ctx->qwery_DWL_cb(ctx, &qry, &answer[0]);
		else z=0;
#else 
		z=0;
		wl_qry=FALSE;
#endif // ENABLE_FIND_WL_SCP

#endif // DEMO
	}
	else  
	{
		//z=DB.Qwery(&qry, &answer[0]);
		if(ctx->db_ctx) z=ctx->db_ctx->qwery_cb(ctx, &qry, &answer[0]);
		else z=0;
	}
#endif
	if(z)
	{	
		for(i=0;i<z;i++)
		{	
			CLEAR_BUFFER;
			l=0;
			DCMAddCommand(buffer,&l,0x00000000,(UINT32)0);
			DCMAddCommand(buffer,&l,0x00020000,(char*)store_sop_uid);
			DCMAddCommand(buffer,&l,0x01000000,(UINT16)0x8020);
			DCMAddCommand(buffer,&l,0x01200000,store_message_id); // message id
			DCMAddCommand(buffer,&l,0x08000000,(UINT16)0x0102); // data set
			DCMAddCommand(buffer,&l,0x09000000,(UINT16)0xFF00); // status
			if(!DCMSendPDV(ctx,l,3)) return FALSE;
			if(!DCMSendPDU(ctx,buffer,l)) return FALSE;
			l=0;
			// send data set
			CLEAR_BUFFER;
			DCMAddCommand(buffer,&l,0x00000000,(UINT32)0); // add group length
			//DCMAddCommand(buffer,&l,0x00520008,(char*)qry.mode.c_str());
			DCMAddCommand(buffer,&l,0x14300004,(char*)qry.mode.c_str());
			for(j=0;j<answer[i].size;j++)
			{	
				DCMAddCommand(buffer,&l,answer[i].element[j].tag,answer[i].element[j].data.c_str()); // status
			}
			ret=2;
			l-=12;
			if(!DCMSendPDV(ctx,l,ret)) return FALSE;
			if(!DCMSendPDU(ctx,buffer+12,l,1)) return FALSE;
		}
	}
	// send ack
	CLEAR_BUFFER;
	l=0;
	DCMAddCommand(buffer,&l,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&l,0x00020000,(char*)store_sop_uid);
	DCMAddCommand(buffer,&l,0x01000000,(UINT16)0x8020);
	DCMAddCommand(buffer,&l,0x01200000,store_message_id); // message id
	DCMAddCommand(buffer,&l,0x08000000,(UINT16)0x0101); // no data set
	DCMAddCommand(buffer,&l,0x09000000,(UINT16)0x0); // status
	if(!DCMSendPDV(ctx,l,3)) return FALSE;
	if(!DCMSendPDU(ctx,buffer,l)) return FALSE;
#ifdef DICOM_SERVER
	{	
		if(!wl_qry) 
		{	
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM сессия #%d.FIND SCP - OK [%s>%s:%s:%s %d записей] <- %s",
				ctx->session_id, qry.mode.c_str(), qry.patient.c_str(), qry.patient_id.c_str(), qry.study_id.c_str(), z, ctx->assotiation_calling_AET);
		}
		else 
		{
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "FIND SCP (WorkList)- OK <- %s", ctx->assotiation_calling_AET);
		}
	}
#endif
	if(ctx->event_exit) SetEvent(ctx->event_exit);
	return TRUE;
}

BOOL DCM_CMD_STORE_SCU(char* buffer, dicom_session_context_t* ctx, image_store_info_t* storage, int mode)
{	
	int l=0, z, i, ret,total,offset;
	int buffer_size;
	PDU_DATA_TF_EX_t* dptr;
	char* bptr=buffer;
	char* sop;
	BOOL retval=TRUE;

#ifdef DICOM_SERVER
	if(ctx->db_ctx && (!ctx->db_ctx->get_storage_info_cb(ctx, storage)))
	{
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
#endif //DICOM_SERVER
	if(storage->sop_UID.size()) sop = (char*)storage->sop_UID.c_str();
	else sop=STORAGE_CT_IMAGE;
	// write store-rq
	CLEAR_BUFFER;
	DCMAddCommand(buffer,&l,0x00000000, (UINT32)0);
	DCMAddCommand(buffer,&l,0x00020000, sop);
	DCMAddCommand(buffer,&l,0x01000000, (UINT16)0x0001);
	DCMAddCommand(buffer,&l,0x01100000, (UINT16)GetU16Odd(ctx)); // message id
	DCMAddCommand(buffer,&l,0x07000000, (UINT16)0x0002); // priority
	if(!mode)
	{
		DCMAddCommand(buffer,&l,0x08000000, (UINT16)0x0001); // data set
		DCMAddCommand(buffer,&l,0x10000000, storage->image_UID.c_str()); // 
	}
	else
	{
		DCMAddCommand(buffer,&l,0x08000000, (UINT16)0x0102); // data set
		DCMAddCommand(buffer,&l,0x10000000, storage->image_UID.c_str()); // 
		DCMAddCommand(buffer,&l,0x10300000, ctx->assotiation_calling_AET);
		DCMAddCommand(buffer,&l,0x10310000, (UINT16)ctx->parent_message_id); // message id
	}
	if(!DCMSendPDV(ctx,l,3)) // last 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	if(!DCMSendPDU(ctx, buffer, l)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// send file
	i=MAX_BUFFER_SEND_SIZE-sizeof(PDU_DATA_TF_EX_t);
	while(!feof(storage->datafile))
	{	
		l=fread(buffer, 1, i, storage->datafile);
		if(l<i) 
		{	
			z=0x02;
		}
		else z=0;
		if(!DCMSendPDV(ctx, l, z)) 
		{	
			retval=FALSE;
			break;
		}
		if(!DCMSendPDU(ctx,buffer,l,1)) 
		{	
			retval=FALSE;
			break;
		}
	}
	// get rq
	memset(buffer,0,512);
	l=0;
	// receive ack
	while(!l)
	{	
		buffer_size=FIRST_PACKET;
		ret=DCMRecvData(ctx,buffer,buffer_size);
		if(ret==SOCKET_ERROR) break;
		dptr=(PDU_DATA_TF_EX_t*)buffer;
		if(dptr->type==0x05) // assoc release rq
		{	
			break;
		}
		if(dptr->type==0x07) // assoc abort
		{	
			retval=FALSE;
			break;
		}
		sb(&dptr->length); // correct endianing
		buffer_size=dptr->length-ret+4+2; 
		if(buffer_size>MAX_BUFFER_SIZE) 
		{	
			retval=FALSE;
			break;
		}
		if(buffer_size<=0) 
		{	
			retval=FALSE;
			break;
		}
		offset=ret;total=0;
		while(total<buffer_size)
		{	
			ret=DCMRecvData(ctx,buffer+total+offset,buffer_size-total);
			total+=ret;
			if(ret==SOCKET_ERROR) 
			{	
				retval=FALSE;
				break;
			}
		}
		//ret=recv(ctx->socket,buffer+ret,buffer_size,0);
		if(ret==SOCKET_ERROR) 
		{	
			retval=FALSE;
			break;
		}
		l=1;
	}
	if(retval && ctx->logger)
	{
		ctx->logger( DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM сессия #%d. STORE_SCU - OK. [UID: %s]", ctx->session_id, storage->image_UID.c_str());
	}
	if(storage->datafile)
	{
		fclose(storage->datafile); // close file handler
		storage->datafile=NULL;
	}
	if(ctx->event_exit) SetEvent(ctx->event_exit);
	return retval;
}

BOOL DCM_CMD_MOVE_SCU(char* buffer, dicom_session_context_t* ctx,char* destination_AET,DCM_QWERY_t* qwery)
{	
	int cmd_len=0,cnt;
	int move_remaining;
	int move_completed;
	int move_failed;
	int move_warning;
	int l,z,last_data=0;
	int buffer_size,total=0;
	DATA_SET_CMD_t* ptr;
	PDU_DATA_TF_EX_t* pdv;
	char* bptr;
	
	CLEAR_BUFFER;
	// qry rq
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00020000,QRY_MOVE_STUDY);
	DCMAddCommand(buffer,&cmd_len,0x01000000,(UINT16)0x0021); // move rq
	DCMAddCommand(buffer,&cmd_len,0x01100000,GetU16Odd(ctx)); // message id
	DCMAddCommand(buffer,&cmd_len,0x06000000,destination_AET);
	DCMAddCommand(buffer,&cmd_len,0x07000000,(UINT16)0x0);
	DCMAddCommand(buffer,&cmd_len,0x08000000,(UINT16)0x0102);
	DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	DCMAddCommand(buffer,&cmd_len,0x00520008,qwery->mode.c_str());// Query Level
	for(cnt=0;cnt<qwery->size;cnt++)
	{	
		DCMAddCommand(buffer,&cmd_len,qwery->element[cnt].tag,qwery->element[cnt].data.c_str());
	}
	if(!DCMSendPDV(ctx,cmd_len,2)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// step2
	cmd_len=0;
	// qwery data
	CLEAR_BUFFER;
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	DCMAddCommand(buffer,&cmd_len,0x00520008,qwery->mode.c_str());// Query Level
	for(cnt=0;cnt<qwery->size;cnt++)
	{	
		DCMAddCommand(buffer,&cmd_len,qwery->element[cnt].tag,qwery->element[cnt].data.c_str());
	}
	if(!DCMSendPDV(ctx,cmd_len,2)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE; // data_set
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// receive ack
	move_remaining=0;
	move_completed=0;
	move_failed=0;
	move_warning=0;
	while(1)
	{	
		buffer_size=DCM_GET_DATA(buffer,ctx);
		if(buffer_size<0) break;
		pdv=(PDU_DATA_TF_EX_t*)buffer;
		// reply analyzer
		for(l=0,last_data=0,bptr=buffer+12;l<buffer_size;)
		{	
			ptr=(DATA_SET_CMD_t*)bptr;
			if(ctx->assotiation_mode&ASS_EXPLICIT) z=ptr->reserved;  
			else z=ptr->val_size;
			if(pdv->tag&0x01) //cmd
			{	
				if(ptr->tag==0x01000000)
				{	
					if(ptr->data!=0x8021) // wrong reply
					{	
						last_data=1;
						break;
					}
					else
					{	
						last_data=0;
					}
				}
			}
			//else
			{	
				switch(ptr->tag)
				{	
					case 0x10200000 : move_remaining=ptr->data;break; // remaining
					case 0x10210000 : move_completed=ptr->data;break; // completed cstore
					case 0x10220000 : move_failed=ptr->data;break; // failed cstore
					case 0x10230000 : move_warning=ptr->data;break;// warning cstore
					case 0x09000000 : if(ptr->data==0x00) last_data=1;break;
				}
			}
			bptr+=z+8;
			l+=z+8;
		}
#ifndef DICOM_SERVER
		if(ctx->event_step) SetEvent(ctx->event_step);
#endif
		if((pdv->tag==0x03) && last_data) break;
	}
	// use external 
	if(ctx->event_exit) SetEvent(ctx->event_exit);
	return TRUE;
}

BOOL DCM_CMD_MOVE_SCP(char* buffer, dicom_session_context_t* ctx)
{	
	int l,z,i,ret,total,offset,error_cnt,len;
	int buffer_size;
	int transfer_syntax=0;
	BOOL data_set=FALSE;
	DATA_SET_CMD_t* ptr;
	DCM_QWERY_t qry;
	PDU_DATA_TF_EX_t* dptr;
	char* bptr=buffer;
	char move_sop_uid[64];
	char remote_aet[32];
	char remote_addr[32];
	char move_instance_uid[64];
	char stg[128];
	int no_send=0;
	std::string rep, tot;
	CWordArray files2send;
	UINT16 move_message_id=0;
	 dicom_session_context_t move_ctx;
	struct sockaddr_in addr;
	std::string sop_UID;
	std::string image_UID;
	image_store_info_t store_info;

	memset(move_sop_uid,0,64);
	memset(move_instance_uid,0,64);
	memset(remote_aet,0,32);
	memset(remote_addr,0,32);
    memset(&move_ctx,0,sizeof( dicom_session_context_t));
	move_ctx.socket=INVALID_SOCKET;
	ClearDCMQwery(&qry);
	len=1024;
	for(l=0,bptr=buffer+12;l<len;)
	{	
		ptr=(DATA_SET_CMD_t*)bptr;
		if(ctx->assotiation_mode&ASS_EXPLICIT) z=ptr->reserved;  
		else z=ptr->val_size;
		if(ptr->tag==0x00000000) len=ptr->data+12;
		else if(ptr->tag==0x00020000) memcpy(move_sop_uid,&ptr->data,z);
		else if(ptr->tag==0x06000000) memcpy(remote_aet,&ptr->data,z);
		else if(ptr->tag==0x01100000) move_message_id=ptr->data;
		else if(ptr->tag==0x10000000) memcpy(move_instance_uid,&ptr->data,z);
		else if(ptr->tag==0x08000000)
		{	if(ptr->data!=0x0101) data_set=TRUE;
		}
		bptr+=ptr->val_size+8;
		l+=ptr->val_size+8;
	}
	//check data set
	if(data_set)
	{	l=0;
		//memset(buffer,0,512);
		CLEAR_BUFFER;
		while(!l)
		{	
			buffer_size=FIRST_PACKET;
			//ret=recv(ctx->socket,buffer,buffer_size,0);
			ret=DCMRecvData(ctx,buffer,buffer_size);
			if(ret==SOCKET_ERROR) break;
			dptr=(PDU_DATA_TF_EX_t*)buffer;
			sb(&dptr->length); // correct endianing
			buffer_size=dptr->length-ret+4+2; 
			if(buffer_size>MAX_BUFFER_SIZE) break;
			if(buffer_size<=0) break;
			offset=ret;total=0;
			while(total<buffer_size)
			{	//ret=recv(ctx->socket,buffer+total+offset,buffer_size-total,0);
				ret=DCMRecvData(ctx,buffer+total+offset,buffer_size-total);
				total+=ret;
				if(ret==SOCKET_ERROR) break;
			}
			if(ret==SOCKET_ERROR) break;
			l=1;
		}
		if(l && (*buffer==0x04))
		{	// analyze query
			for(l=0,bptr=buffer+12;l<buffer_size;)
			{	ptr=(DATA_SET_CMD_t*)bptr;
				if(ctx->assotiation_mode&ASS_EXPLICIT) z=ptr->reserved;  
				else z=ptr->val_size;
				if(z==0xFFFF) z=0; // skip sequence tags
				if(qry.size<MAX_QRY)	
				{	if(z>127) z=127;
					memset(stg,0,128);
					memcpy(stg,&ptr->data,z);
					switch(ptr->tag)
					{	case 0x00010000 :	break;
						case 0x00001000 :	break;
						case 0x00008000 :	break;
						case 0x00520008 :	qry.mode=stg; break;
											DCM_INIT_QRY_TAGS(&qry,move_sop_uid);break; 
						default :	DCM_SET_QRY_TAG(&qry,ptr->tag,stg);break;
					}
				}
				bptr+=z+8;
				l+=z+8;
			}
		}
	}
	files2send.RemoveAll();
#ifdef DICOM_SERVER
	//DB.GetFiles(&qry, &files2send, &transfer_syntax);
	if(ctx->db_ctx) ctx->db_ctx->get_files_cb(ctx, &qry, &files2send, &transfer_syntax);
#endif
	// start new assoc
	move_ctx.remote_aet=remote_aet;
	l=sizeof(addr);
	ret=getpeername(ctx->socket,(LPSOCKADDR)&addr,&l);  
	if(ret==SOCKET_ERROR) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	//ctx->remote_port=addr.sin_port;
	if(!strlen(remote_aet)) memcpy(remote_aet,ctx->assotiation_calling_AET,16);
	//move_ctx.remote_port=DB.FindProviderPort(remote_aet);
	//move_ctx.remote_addr=DB.FindProviderAddr(remote_aet);
	if(ctx->db_ctx) move_ctx.remote_port=ctx->db_ctx->partner_find_port_cb(ctx, remote_aet);
	if(ctx->db_ctx) move_ctx.remote_addr=ctx->db_ctx->partner_find_addr_cb(ctx, remote_aet);
	move_ctx.parent_message_id=move_message_id;
	memcpy(move_ctx.assotiation_calling_AET, Get_own_AET(ctx), Get_own_AET_len(ctx));
	if(move_ctx.remote_addr==NULL) 
	{	
		if(ctx->logger) ctx->logger( DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. MOVE SCP Ошибка - невозможно определить адрес", ctx->session_id);
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	CLEAR_BUFFER;
	total=files2send.GetCount();
	//if(!CreateAssociacion(buffer,&move_ctx,NULL,transfer_syntax)) 
	if(!CreateAssociacion(buffer,&move_ctx,NULL,0)) 
	{	
		if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. MOVE SCP Ошибка - невозможно открыть вторичную ассоциацию [pid:%d] (%s|%s|%d).",
			ctx->session_id,move_ctx.association_PID,move_ctx.remote_aet,move_ctx.remote_addr,move_ctx.remote_port);
		//if(ctx->event_exit) SetEvent(ctx->event_exit);
		//return FALSE;
		no_send=1;
		error_cnt=total;
	}
	else
	{	
		if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM сессия #%d. Вторичная ассоциация установлена [pid:%d] (%s|%s|%d).",
			ctx->session_id,move_ctx.association_PID,move_ctx.remote_aet,move_ctx.remote_addr,move_ctx.remote_port);
		error_cnt=0;
	}
	i=0;
	for(;(!no_send) && (i<files2send.GetCount());i++) 
	{	
		// send data
		CLEAR_BUFFER;
		store_info.image_id=files2send[i];
		if(!DCM_CMD_STORE_SCU( buffer, &move_ctx, &store_info, 1)) 
		{
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. MOVE SCP Ошибка при вызове STORE_SCU [%d]", ctx->session_id, i);
			error_cnt++;
		}
		// send ack
		CLEAR_BUFFER;
		l=0;
		DCMAddCommand(buffer, &l,0x00000000,(UINT32)0);
		DCMAddCommand(buffer, &l,0x00020000,&move_sop_uid[0]);
		DCMAddCommand(buffer, &l,0x01000000,(UINT16)0x8021);
		DCMAddCommand(buffer,&l,0x01200000,(UINT16)move_message_id); // message id
		DCMAddCommand(buffer,&l,0x08000000,(UINT16)0x0101); // no data set
		DCMAddCommand(buffer,&l,0x09000000,(UINT16)0xFF00); // status
		DCMAddCommand(buffer,&l,0x10200000,(UINT16)total); // remaining
		DCMAddCommand(buffer,&l,0x10210000,(UINT16)i); // completed cstore
		DCMAddCommand(buffer,&l,0x10220000,(UINT16)0); // failed cstore
		DCMAddCommand(buffer,&l,0x10230000,(UINT16)0); // warning cstore
		//if(l&0x01) l++;
		if(!DCMSendPDV(ctx,l,3)) 
		{	
			if(ctx->event_exit) SetEvent(ctx->event_exit);
			if(ctx->logger) ctx->logger( DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. MOVE SCP Ошибка", ctx->session_id);
			return FALSE;
		}
		if(!DCMSendPDU(ctx,buffer,l)) 
		{	
			if(ctx->event_exit) SetEvent(ctx->event_exit);
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. MOVE SCP Ошибка", ctx->session_id);
			return FALSE;
		}
		total--;
		if(ctx->event_step) SetEvent(ctx->event_step);
	}
	if(!no_send)
	{	
		// close store ass
		DCMEndAssociationRq(&move_ctx,buffer);
		DCMEndAssociationGetRply(&move_ctx,buffer);
		DCMCloseConnection(&move_ctx,DCM_SERVER);
	}
	// send final ack
	// send ack
	CLEAR_BUFFER;
	l=0;
	DCMAddCommand(buffer,&l,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&l,0x00020000,&move_sop_uid[0]);
	DCMAddCommand(buffer,&l,0x01000000,(UINT16)0x8021);
	DCMAddCommand(buffer,&l,0x01200000,(UINT16)move_message_id); // message id
	DCMAddCommand(buffer,&l,0x08000000,(UINT16)0x0101); // no data set
	DCMAddCommand(buffer,&l,0x09000000,(UINT16)0x00); // status
	DCMAddCommand(buffer,&l,0x10200000,(UINT16)0); // remaining
	DCMAddCommand(buffer,&l,0x10210000,(UINT16)i); // completed cstore
	DCMAddCommand(buffer,&l,0x10220000,(UINT16)error_cnt); // failed cstore
	DCMAddCommand(buffer,&l,0x10230000,(UINT16)0); // warning cstore
	if(!DCMSendPDV(ctx,l,3)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. MOVE SCP Ошибка", ctx->session_id);
		return FALSE;
	}
	if(!DCMSendPDU(ctx,buffer,l)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. MOVE SCP Ошибка", ctx->session_id);
		return FALSE;
	}
#ifdef DICOM_SERVER
	if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "[%s>%s:%s:%s:%s] %d элементов, %d ошибок", qry.mode.c_str(), 
		qry.patient.c_str(), qry.patient_id.c_str(), qry.study_id.c_str(),
		qry.image_id.c_str(), files2send.GetCount(), error_cnt);
#endif
	if(ctx->event_exit) SetEvent(ctx->event_exit);
	CLEAR_BUFFER;
	return TRUE;
}

BOOL DCM_CMD_CANCEL_SCP(char* buffer, dicom_session_context_t* ctx)
{	
	return TRUE;
}

BOOL DCM_CMD_GET_SCP(char* buffer, dicom_session_context_t* ctx)
{	
	int l,z;
	BOOL data_set=FALSE;
	DATA_SET_CMD_t* ptr;
	DCM_QWERY_t qry;
	char* bptr=buffer;
	char get_sop_uid[64];
	UINT16 get_message_id=0;
	
	memset(get_sop_uid,0,64);
	for(l=0,bptr=buffer+12;l<1024;)
	{	
		ptr=(DATA_SET_CMD_t*)bptr;
		z=ptr->val_size;
		if(ptr->tag==0x00020000) memcpy(get_sop_uid,&ptr->data,z);
		else if(ptr->tag==0x01100000) get_message_id=ptr->data;
		bptr+=ptr->val_size+8;
		l+=ptr->val_size+8;
	}
	// send ack
	CLEAR_BUFFER;
	l=0;
	DCMAddCommand(buffer,&l,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&l,0x00020000,&get_sop_uid[0]);
	DCMAddCommand(buffer,&l,0x01000000,(UINT16)0x8010);
	DCMAddCommand(buffer,&l,0x01200000,(UINT16)get_message_id); // message id
	DCMAddCommand(buffer,&l,0x08000000,(UINT16)0x0101); // no data set
	DCMAddCommand(buffer,&l,0x09000000,(UINT16)0x0000); // status
	DCMAddCommand(buffer,&l,0x10200000,(UINT16)0); // remaining
	DCMAddCommand(buffer,&l,0x10210000,(UINT16)0); // completed cstore
	DCMAddCommand(buffer,&l,0x10220000,(UINT16)0); // failed cstore
	DCMAddCommand(buffer,&l,0x10230000,(UINT16)0); // warning cstore
	//if(l&0x01) l++;
	if(!DCMSendPDV(ctx,l,3)) return FALSE;
	if(!DCMSendPDU(ctx,buffer,l)) return FALSE;
#ifdef DICOM_SERVER
	if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_WARNING, "GET SCP - n\a");
#endif
	if(ctx->event_exit) SetEvent(ctx->event_exit);
	return TRUE;
}

BOOL DCM_CMD_BAD( char* buffer,  dicom_session_context_t* ctx, int command)
{ 
#ifdef DICOM_SERVER
	if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM сессия #%d. Неизвестная или не поддерживаемая команда : %d", 
		ctx->session_id, command);
#endif
	return TRUE; 
}

BOOL AnalyzeCommand(char* buffer, dicom_session_context_t* ctx)
{	
	DATA_SET_CMD_t* ptr;
	char* bptr=buffer;
	UINT16 l;
	int x_size;
	int DICOM_COMMAND=0;
	BOOL retval=TRUE;
	
	if(*bptr==0x04) bptr+=12;
	ptr=(DATA_SET_CMD_t*)bptr;
	x_size=ptr->data;
	for(l=0;l<=x_size;)
	{	
		ptr=(DATA_SET_CMD_t*)bptr;
		if(ptr->tag==0x01000000) DICOM_COMMAND=ptr->data;
		bptr+=ptr->val_size+8;
		l+=ptr->val_size+8;
	}
	switch(DICOM_COMMAND)
	{	
		case 0x0030 : DCM_CMD_ECHO_SCP( buffer, ctx); break;
#ifdef ENABLE_STORE_SCP
		case 0x0001 : DCM_CMD_STORE_SCP( buffer, ctx); break;
#endif // ENABLE_STORE_SCP
#ifdef ENABLE_FIND_SCP
		case 0x0020 : DCM_CMD_FIND_SCP( buffer, ctx); break;
#endif // ENABLE_FIND_SCP
#ifdef ENABLE_MOVE_SCP
		case 0x0021 : DCM_CMD_MOVE_SCP( buffer, ctx); break;
#endif // ENABLE_MOVE_SCP
		case 0x0FFF : DCM_CMD_CANCEL_SCP( buffer, ctx); break;
#ifdef ENABLE_GET_SCP
		case 0x0010 : DCM_CMD_GET_SCP( buffer, ctx); break;
#endif // ENABLE_GET_SCP
		default :	DCM_CMD_BAD( buffer, ctx, DICOM_COMMAND);
					retval=FALSE; break;
	}
	return retval;
}

void DCMRemoveSession( dicom_session_context_t* ctx)
{
	dicom_server_context_t* pctx=NULL;
	int i;

	if(ctx==NULL) return;
	pctx=(dicom_server_context_t*)ctx->parent;
	if(pctx==NULL) return;
	CSingleLock m_lock(pctx->srv_lock);
	m_lock.Lock();
	for(i=0; i<pctx->sessions->GetCount(); i++)
	{
		if(pctx->sessions->GetAt(i)==ctx)
		{
			if(ctx->db_ctx) ctx->db_ctx->close_connection_cb(ctx);
			free(pctx->sessions->GetAt(i));
			pctx->sessions->RemoveAt(i);
		}
	}
	m_lock.Unlock();
}

DWORD WINAPI DCMSessionProc(PVOID pParameter)
{	
	dicom_session_context_t* ctx;
	ASSOCIATE_RQ_t* arq;
	PDU_DATA_TF_EX_t*  cmd;
	int total,offset;
	char session_buffer[MAX_BUFFER_SIZE];
	int buffer_size;
	int ret,err;
	int ass_done=0;
	byte val;
	
	ctx=( dicom_session_context_t*)pParameter;
	while(1)
	{	
		// start association
		buffer_size=FIRST_PACKET;
		//ret=recv(ctx->socket,session_buffer,buffer_size,0);
		ret=DCMRecvData(ctx,session_buffer,buffer_size);
		if(ret==SOCKET_ERROR) 
		{	
			err=WSAGetLastError();
			break;
		}
		arq=(ASSOCIATE_RQ_t*)session_buffer;
		sb(&arq->length); // correct endianing
		buffer_size=arq->length-ret+4+2; //??
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
			//ret=recv(ctx->socket,session_buffer+total+offset,buffer_size-total,0);
			ret=DCMRecvData(ctx,session_buffer+total+offset,buffer_size-total);
			total+=ret;
			if(ret==SOCKET_ERROR) 
			{	
				break;
			}
		}
		if(ret==SOCKET_ERROR) 
		{	
			break;
		}
		// analyze association
		if(!StartAssociation(session_buffer,arq->length+4,ctx)) break;
		ass_done=1;
		break;
	}
	while(ass_done)
	{	
		buffer_size=FIRST_PACKET;
		//ret=recv(ctx->socket,session_buffer,buffer_size,0);
		ret=DCMRecvData(ctx,session_buffer,buffer_size);
		if(ret==SOCKET_ERROR) 
		{	
			break;
		}
		if(session_buffer[0]==0x07) // assoc abort
		{	
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM session #%d aborted by peer", ctx->session_id);
			break;
		}
		if(session_buffer[0]==0x05) // assoc release
		{	
			if(!DCMEndAssociationRply(ctx,session_buffer)) 
			{	
				break;
			}
			// close session
			if(ctx->status==SESSION_STOPPED) 
			{	
				break;
			}
		}
		cmd=(PDU_DATA_TF_EX_t*)session_buffer;
		sb(&cmd->length); // correct endianing
		buffer_size=cmd->length-ret+4+2; 
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
			ret=DCMRecvData(ctx,session_buffer+total+offset,buffer_size-total);
			total+=ret;
			if(ret==SOCKET_ERROR) 
			{	
				break;
			}
		}
		if(ret==SOCKET_ERROR) 
		{	break;
		}
		if(cmd->type!=0x04) 
		{	
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM session #%d aborted. Wrong data.", ctx->session_id);
			break;
		}
		// handle ass endianing & etc
		ctx->association_PID=*(session_buffer+10);
		ctx->assotiation_mode=0;
		val=ctx->association[(ctx->association_PID>>1)&0x7F].type;
		switch(val)
		{	
			case 0 : ctx->assotiation_mode=0;break;
			case 1 : 
			case 2 : ctx->assotiation_mode=ASS_EXPLICIT;break;
			case 3 : ctx->assotiation_mode=ASS_EXPLICIT|ASS_BIGENDIAN;break;
			default : ctx->assotiation_mode=ASS_EXPLICIT;break;
		}
		if(!AnalyzeCommand(session_buffer, ctx)) break;
	}
	DCMCloseConnection(ctx, DCM_SERVER);
	// log
	if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM session #%d closed", ctx->session_id);
	// free session
	DCMRemoveSession(ctx);
	return 0;
}

void ResetUID( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) return;
	ctx->U_8=0;
	ctx->U_16=0;
	ctx->U_32=0;
}

int8_t GetU8( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) return 0;
	ctx->U_8+=1;
	return ctx->U_8;
}

UINT16 GetU16( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) return 0;
	ctx->U_16+=1;
	return ctx->U_16;
}

UINT16 GetU16Odd( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) return 0;
	ctx->U_16+=1;
	if(ctx->U_16&0x01) return ctx->U_16;
	else ctx->U_16+=1;
	return ctx->U_16;
}

UINT32 GetU32( dicom_session_context_t* ctx)
{	
	if(ctx==NULL) return 0;
	ctx->U_32+=1;
	return ctx->U_32;
}

BOOL CreateAssociacion(char* buffer, dicom_session_context_t* ctx,char* p_context,int t_syntax)
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
	hndl=CreateEvent(NULL,TRUE,FALSE,NULL);
	// create socket
	// connect
	if((ctx->socket==INVALID_SOCKET) || (ctx->socket==NULL))
	{	
		ctx->socket=socket(AF_INET, SOCK_STREAM,IPPROTO_TCP);
		if(ctx->socket==INVALID_SOCKET)
		{	
			//AfxMessageBox("Невозможно создать сокет отправителя.");
			ctx->err_code = 1;
			return FALSE;
		}
		BOOL reuse=TRUE;
		setsockopt(ctx->socket,SOL_SOCKET,SO_REUSEADDR,(char*)&reuse,sizeof(BOOL));
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
			return FALSE;
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
		return FALSE;
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
		return TRUE;
	else 
		return FALSE;
}

 dicom_session_context_t* GetNewSession(dicom_server_context_t* pctx)
{	
	 dicom_session_context_t* ctx=NULL;
	CSingleLock m_lock(pctx->srv_lock);

	m_lock.Lock();
	if(pctx->sessions->GetCount()>=MAX_DICOM_SESSIONS) 
	{
		if(pctx->logger)
		{
			if(pctx->logger) pctx->logger(DCM_LOG_SYSTEM_EVENT, DCM_LOG_ERROR,
				"Превышено допустимое количество сессий (%d).", MAX_DICOM_SESSIONS);
		}
	}
	else
	{
		ctx=( dicom_session_context_t*)malloc(sizeof( dicom_session_context_t));
		if(ctx)
		{
			// clear session
			ctx->status=SESSION_STOPPED;
			ctx->socket=INVALID_SOCKET;
			ctx->session=NULL;
			ctx->session_id=0;
			ctx->timeout=pctx->timeout;
			ctx->parent=(void*)pctx;
			ctx->db_ctx=pctx->db_ctx;
			if(ctx->db_ctx) ctx->db_ctx->set_connection_cb(ctx);
			// logger
			//ctx->log_level=pctx->log_level;
			//ctx->log_mode=pctx->log_mode;
			//ctx->log_subsys=pctx->log_subsys;
			ctx->logger=pctx->logger;
			ctx->dumper=pctx->dumper;
			// add to array
			pctx->sessions->Add(ctx);
		}
	}
	m_lock.Unlock();
	return ctx;
}

DWORD WINAPI DCMServerProc(PVOID pParameter)
{	
	SOCKET session_socket;
	DWORD dwThreadId=0;
	dicom_server_context_t* pctx=(dicom_server_context_t*)pParameter;
	 dicom_session_context_t* ctx=NULL;

	if(pctx==NULL) return 0;
	pctx->state=SESSION_STARTED;
	while(1)
	{	
		session_socket=(SOCKET)SOCKET_ERROR;
		while(session_socket==SOCKET_ERROR)
		{	
			session_socket=accept(pctx->socket, NULL, NULL );
			if(session_socket!=INVALID_SOCKET)
			{	
				// start new session
				ctx=GetNewSession(pctx);
				if((pctx->state==SESSION_STOPPED) || (ctx==NULL))
				{	
					shutdown(session_socket, SD_BOTH);
					closesocket(session_socket);
					if(pctx->logger) pctx->logger(DCM_LOG_SYSTEM_EVENT, DCM_LOG_OK,
						"Сервер остановлен. Невозможно принять новую сессию.");
					break;
				}
				ctx->socket=session_socket;
				ctx->status=SESSION_STARTED;
				ctx->session_id=(unsigned int)ctx;
				// create process
				ctx->session=CreateThread(NULL, 0, DCMSessionProc, (void*)ctx, NULL, &dwThreadId);
				if(ctx->session==NULL) 
				{	
					ctx->status=SESSION_STOPPED;
					if(pctx->logger) pctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "Ошибка открытия DICOM сессии");
				}
				else 
				{	
					if(pctx->logger) pctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM session #%d open", ctx->session_id);
				}
			}
		}
	}
	return 0;
}

// start server
BOOL DICOMServerStart(dicom_server_context_t* pctx)
{	
	int ret;
	BOOL retval=FALSE;
#ifdef DICOM_SERVER
	nic_info_t* info;
#endif
	DWORD dwThreadId=0;
	struct sockaddr_in addr;  
	// set options
	BOOL reuse=TRUE;
	
	if(pctx==NULL) return retval;
	if(pctx->state==SESSION_STARTED) return TRUE;
	if(pctx->main_thread!=NULL)
	{
		pctx->state=SESSION_STARTED;
		return TRUE;
	}
	pctx->socket=socket(AF_INET, SOCK_STREAM,IPPROTO_TCP);
	if(pctx->socket==INVALID_SOCKET)
	{	
		//AfxMessageBox("Невозможно создать сокет сервера.");
		if(pctx->logger) pctx->logger(DCM_LOG_SYSTEM_EVENT, DCM_LOG_ERROR, "Невозможно создать сокет сервера.");
		return FALSE;
	}
	setsockopt(pctx->socket, SOL_SOCKET, SO_REUSEADDR,(char*)&reuse, sizeof(BOOL));
	addr.sin_family=AF_INET;
	addr.sin_port=htons((unsigned short)pctx->port);
#ifdef DICOM_SERVER
	if(pctx->nic_idx==0) addr.sin_addr.s_addr=htonl(INADDR_ANY);
	else
	{	
		get_nic_all();
		info=get_nic_info(pctx->nic_idx-1);
		if(info==NULL) addr.sin_addr.s_addr=htonl(INADDR_ANY);
		else addr.sin_addr.s_addr=get_inet_addr( (char*)info->ip_addr.c_str());
	}
#else
	addr.sin_addr.s_addr=htonl(INADDR_ANY);
#endif //DICOM_SERVER
	ret=bind(pctx->socket, (sockaddr*)&addr, sizeof(sockaddr));
	ret=listen(pctx->socket, SOMAXCONN);
	if(ret==SOCKET_ERROR) return retval;
	// start process
	pctx->main_thread=CreateThread(NULL, 0, DCMServerProc, pctx, 0, &dwThreadId);
	if(pctx->main_thread==NULL) return FALSE;
	return TRUE;
}

dicom_server_context_t* DICOMServerCreate(DICOM_PROVIDER_t* settings, dcm_logger_cb logger_cb,
										  dcm_dumper_cb dumper_cb, 
										  db_handlers_t* db_ctx,
										  int timeout, 
										  int nic_idx/* =0 */)
{
	dicom_server_context_t* ctx=NULL;

	if(settings==NULL) return ctx;
	if(db_ctx==NULL) return ctx;
	ctx=(dicom_server_context_t*)malloc(sizeof(dicom_server_context_t));
	if(ctx==NULL) return ctx;
	memset(ctx, 0, sizeof(dicom_server_context_t));
	ctx->sessions=new CPtrArray();
	ctx->srv_lock=new CCriticalSection();
	strcpy(ctx->aet, settings->AET.c_str());
	ctx->port=settings->port;
	ctx->nic_idx=nic_idx;
	ctx->logger=logger_cb;
	ctx->dumper=dumper_cb;
	ctx->timeout=timeout;
	ctx->own_guid=OWN_GUID;
	ctx->sessions->RemoveAll();
	ctx->db_ctx=db_ctx;
	ctx->db_conn=NULL; //!!!!!!!!!!!
	//ctx->log_level=log_level;
	//ctx->log_mode=log_mode;
	//ctx->log_subsys=log_subsys;
	return ctx;
}

BOOL DICOMServerClose(dicom_server_context_t* pctx)
{
	BOOL retval=FALSE;
	int i;

	if(pctx==NULL) return retval;
	if(pctx->sessions->GetCount())
	{
		for(i=0; i< pctx->sessions->GetCount(); i++) 
		{
			// stop thread and free contexts
			free(pctx->sessions->GetAt(i));
		}
	}
	delete pctx->sessions;
	delete pctx->srv_lock;
	free(pctx);
	retval=TRUE;
	return retval;
}

BOOL DICOMServerStop(dicom_server_context_t* pctx)
{
	BOOL retval=FALSE;

	if(pctx==NULL) return retval;
	pctx->state=SESSION_STOPPED;
	retval=TRUE;
	return retval;
}

int GetSOPId(char* sop)
{	
	int i;
	int retval=7; // CT image
	
	if(sop==0) return retval;
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

// clean data subroutines
// 
void clear_data(PATIENT_DATA_t* pd)
{
	pd->PatientID.clear();
	pd->name.clear();
	pd->BirthDate.clear();
	pd->Sex.clear();
	pd->Age.clear();
	pd->GID=0;
	pd->tm_created=0;
	pd->tm_last_access=0;
	pd->state=0;
}

void clear_data(EXAM_DATA_t* ed)
{
	ed->ExamID.clear();
	ed->name.clear();
	ed->StudyDate.clear();
	ed->StudyTime.clear();
	ed->AccessionNumber.clear();
	ed->Modality.clear();
	ed->Description.clear();
	ed->BodyPart.clear();
	ed->GID=0;
	ed->PatientID=0;
	ed->tm_created=0;
	ed->tm_last_access=0;
	ed->state=0;
}

void clear_data(SERIES_DATA_t* sd)
{
	sd->SeriesID.clear();
	sd->name.clear();
	sd->Description.clear();
	sd->BodyPart.clear();
	sd->Number.clear();
	sd->GID=0;
	sd->ExamID=0;
	sd->tag=0;
	sd->tm_created=0;
	sd->tm_last_access=0;
	sd->state=0;
}

void clear_data(IMAGE_DATA_t* imd)
{
	imd->ImageID.clear();;
	imd->name.clear();;
	imd->GID=0;
	imd->SeriesID=0;
	imd->path.clear();;
	imd->SopUID.clear();;
	imd->tm_created=0;
	imd->tm_last_access=0;
	imd->state=0;
}

void clear_data(DICOM_PROVIDER_t* dp)
{
	dp->name.clear();
	dp->AET.clear();
	dp->addr.clear();
	dp->port=104;
	dp->save_to_arc=0;
}

