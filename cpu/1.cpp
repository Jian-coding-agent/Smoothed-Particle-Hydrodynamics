#include <fstream>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>
#include <stdlib.h>

using namespace std;

const int num=237298;   //水流粒子+墙粒子的总和  total1+nvirt
//definition of all variables
double hsml,r,pi;//kernel function variables
double driac;
int countiac[num],ntotal,niac;
std::vector<int> pair_i;
std::vector<int> pair_j;
double x[3][num];
int integ;
int i,j,k,d;
int m,n,mp,np,vir_mp;
double c;
double x0,x1,dx0,dx1,vir_x1,vir_dx;
double vx[3][num],rho[num],mass[num],p[num],itype[num];
double viscos[3][num];
double vecn[3][num],sumvecn[3][num];
double SurF[num];
double SurFD[num];

double X0[3][num];
double rho0[num];
double a11[num],a12[num],a13[num],a21[num],a22[num],a23[num],a31[num],a32[num],a33[num];
double A11[num],A12[num],A13[num],A21[num],A22[num],A23[num],A31[num],A32[num],A33[num];
double F11[num],F12[num],F13[num],F21[num],F22[num],F23[num],F31[num],F32[num],F33[num];
double DisGrad11[num],DisGrad12[num],DisGrad13[num],DisGrad21[num],DisGrad22[num],DisGrad23[num],DisGrad31[num],DisGrad32[num],DisGrad33[num];
double GLStrain11[num],GLStrain12[num],GLStrain13[num],GLStrain21[num],GLStrain22[num],GLStrain23[num],GLStrain31[num],GLStrain32[num],GLStrain33[num];
double Pia11[num],Pia12[num],Pia13[num],Pia21[num],Pia22[num],Pia23[num],Pia31[num],Pia32[num],Pia33[num];
double PIJW[3][num];
double SForce[3][num];
double Fni11[num],Fni12[num],Fni13[num],Fni21[num],Fni22[num],Fni23[num],Fni31[num],Fni32[num],Fni33[num];
double FniT11[num],FniT12[num],FniT13[num],FniT21[num],FniT22[num],FniT23[num],FniT31[num],FniT32[num],FniT33[num];
double lamd;
double muyou;

void GraCorrec();
void GLStrain();


//数值耗散变量
double DFdt11[num],DFdt12[num],DFdt13[num],DFdt21[num],DFdt22[num],DFdt23[num],DFdt31[num],DFdt32[num],DFdt33[num];
double DFdtB11[num],DFdtB12[num],DFdtB13[num],DFdtB21[num],DFdtB22[num],DFdtB23[num],DFdtB31[num],DFdtB32[num],DFdtB33[num];
double SD11[num],SD12[num],SD13[num],SD21[num],SD22[num],SD23[num],SD31[num],SD32[num],SD33[num];
double PD11[num],PD12[num],PD13[num],PD21[num],PD22[num],PD23[num],PD31[num],PD32[num],PD33[num];

double drhodt[num];
double dvxdt[3][num];
double AveV[3][num];
int nvirt;
double dt;

//definition of all functions
double kernel(double);
double dwdx(double,double);
double dwdr(double);
double max(double,double);
double min(double,double);

void init_grid();
void direct_find1();
void direct_find();
void continue_density();
double MaxV;
double c0 = 48.0;
int itime;
double eta = 0.0016;
int SolNum;

struct stMesh {
	double x, y, z;
	stMesh* parent;
	stMesh* children[8];
	double radio;
	int ptIndex;
	int level;
	int id;
};
void NorVector();

//	所有的网格
std::vector<stMesh*> meshes;
int meshCount = 0;

//	每个点对应的网格
//	std::vector<stMesh*> ptMeshs;

stMesh* createMeshes();

// ============================ [OPT-OMP] ============================
// Per-thread private copies of every accumulator written by the two pair
// loops; they are zeroed per call and summed into the globals afterwards.
static double *g_omp_buf = 0;
static int     g_omp_nth = 0;
static const int g_omp_narr = 35;
// 0:drhodt 1-3:dvxdt 4-6:viscos 7:SurF 8-10:AveV
// 11-19:F11..F33 20-28:DFdt11..DFdt33 29-31:PIJW 32-34:SForce
#define OMPBUF(t, n)  (g_omp_buf + (((size_t)(t) * g_omp_narr) + (n)) * (size_t)num)

void omp_init()
{
	int nth = 1;
#ifdef _OPENMP
	const char* pT = getenv("SPH_THREADS");
	if(pT != 0){ int n = atoi(pT); if(n > 0){ omp_set_num_threads(n); } }
	nth = omp_get_max_threads();
#else
	printf("[OMP] built without /openmp, running serial\n");
#endif
	if(nth < 1){ nth = 1; }
	if(nth > 32){ nth = 32; }
	if(g_omp_buf != 0){ delete [] g_omp_buf; }
	g_omp_buf = new double[(size_t)nth * (size_t)g_omp_narr * (size_t)num]();
	g_omp_nth = nth;
	printf("[OMP] threads = %d, accumulate buffer = %.1f MB\n", nth,
		(double)((size_t)nth * (size_t)g_omp_narr * (size_t)num * sizeof(double)) / 1048576.0);
}

int main()
{
	char sResult[100];
	memset(sResult, 0, 100);

	ifstream fin("input.dat");
	int idx;
	for( idx = 0; idx < num; ++idx )
	{
		fin>>x[0][idx]>>x[1][idx]>>x[2][idx]>>p[idx]>>vx[0][idx]>>vx[1][idx]>>vx[2][idx]>>rho[idx]>>mass[idx]>>itype[idx]>>X0[0][idx]>>X0[1][idx]>>X0[2][idx]>>rho0[idx];
		if(rho[idx]<960){rho[idx]=960;}
	}
	fin.close();

	fstream outfile1,outfile2;
	lamd = 13333333;//3.0;
	muyou = 2222222;//0.56;
	pi = 3.141592653589793;
	dt = 0.00001;
	hsml = 0.01;
    ntotal = 101250;
	SolNum = 60320;
	MaxV = 30.0;

	//input();
	//virt_particle();
	init_grid();
	omp_init();
	NorVector();
    GraCorrec();
	// [VS2026] step cap via environment variable SPH_MAXSTEP (unset = original 10000001)
	int iMaxStep = 10000001;
	const char* pEnvMaxStep = getenv("SPH_MAXSTEP");
	if(pEnvMaxStep != 0){ iMaxStep = atoi(pEnvMaxStep); }
	for(itime=0; itime<iMaxStep; itime++)
	{
		direct_find();
		continue_density();

        for(i=0;i<ntotal;i++)
		{
	        rho[i]   = rho[i] + dt * drhodt[i];
			if(rho[i]<960)
			{rho[i]=960;}
            vx[0][i] = vx[0][i] + dt * (dvxdt[0][i]+viscos[0][i])+dt*SForce[0][i];
            vx[1][i] = vx[1][i] + dt * (dvxdt[1][i]+viscos[1][i])+dt*SForce[1][i];
            vx[2][i] = vx[2][i] + dt * (dvxdt[2][i]+viscos[2][i] - 9.8)+dt*SForce[2][i];

			double absv = sqrt(vx[0][i]*vx[0][i]+vx[1][i]*vx[1][i]+vx[2][i]*vx[2][i]+0.00001);
			if(absv>MaxV)
			{
				vx[0][i] = MaxV*vx[0][i]/absv;
				vx[1][i] = MaxV*vx[1][i]/absv;
				vx[2][i] = MaxV*vx[2][i]/absv;
			}
		}
		for(i=ntotal;i<ntotal+SolNum;i++)//速度
		{
			vx[0][i] = vx[0][i]+dt*PIJW[0][i]+dt*SForce[0][i];
			vx[1][i] = vx[1][i]+dt*PIJW[1][i]+dt*SForce[1][i];
			vx[2][i] = vx[2][i]+dt*PIJW[2][i]+dt*SForce[2][i]-dt*9.8;

			if(x[2][i]<=0.01 || x[1][i]<=0.01 || x[1][i]>=0.49)
			{
				vx[0][i]=0.0; vx[1][i]=0.0; vx[2][i]=0.0;
			}
		}
		//GLStrain();
        for(i=0;i<num;i++)
		{
            x[0][i]  = x[0][i] + dt * vx[0][i];//(vx[0][i]+0.5*AveV[0][i]);
            x[1][i]  = x[1][i] + dt * vx[1][i];//(vx[1][i]+0.5*AveV[1][i]);
            x[2][i]  = x[2][i] + dt * vx[2][i];//(vx[2][i]+0.5*AveV[2][i]);
		}

		if(itime%50 == 0)
		{
			cout<<"T= "<<itime/50<<"	"<<"粒子对="<<niac<<endl; 
			sprintf(sResult, "result%d.dat", itime/50);
			outfile1.open(sResult, ios::out);

	        for(i=0;i<num;i++)//
			{
				double ppp = 0;
				if(i<ntotal){ppp=p[i];}else{ppp = Pia11[i]+Pia22[i]+Pia33[i];}
				outfile1<<x[0][i]<<"	"<<x[1][i]<<"	"<<x[2][i]<<"	"<<ppp<<"	"<<vx[0][i]<<"	"<<vx[1][i]<<"	"<<vx[2][i]<<"	"
						<<rho[i]<<"	"<<mass[i]<<"	"<<itype[i]<<"	"<<X0[0][i]<<"	"<<X0[1][i]<<"	"<<X0[2][i]<<"	"<<rho0[i]<<endl;
			}
			outfile1.close();
		}
	}

	return 0;
}

//核函数(已校验)
double kernel(double dr)
{
    double h = 1.5*hsml;
	double q=dr/h;
	double coe = 21.0/(256.0*h*h*h*pi);
	if(q>=0.0 && q<=2.0)
	{
		return coe*(1.0+2.0*q)*(2.0-q)*(2.0-q)*(2.0-q)*(2.0-q);
	}
	else
	{
		return 0.0;
	}
}
double dwdx(double dx, double dr)
{
	double h = 1.5*hsml;
	double q=dr/h;
	double coe = 42.0/(256.0*h*h*h*h*pi);
	if(q>=0.0 && q<=2.0)
	{
		return coe*((2.0-q)*(2.0-q)*(2.0-q)*(2.0-q)-2.0*(1+2.0*q)*(2.0-q)*(2.0-q)*(2.0-q))*(dx/dr);
	}
	else
	{
		return 0.0;
	}
}
double dwdr(double dr)
{
	double h = 1.5*hsml;
	double q=dr/h;
	double coe = 42.0/(256.0*h*h*h*h*pi);
	if(q>=0.0 && q<=2.0)
	{
		return coe*((2.0-q)*(2.0-q)*(2.0-q)*(2.0-q)-2.0*(1+2.0*q)*(2.0-q)*(2.0-q)*(2.0-q));
	}
	else
	{
		return 0.0;
	}
}

stMesh* createMesh()
{
	stMesh* mesh = 0;
	if( meshCount >= (int)meshes.size() )
	{
		mesh = new stMesh();
		meshes.push_back( mesh );
	}
	else 
	{
		mesh = meshes[ meshCount ];
	}
	mesh->parent = 0;
	memset( mesh->children, 0, sizeof( stMesh* ) * 8 );
	mesh->level = -1;
	mesh->x = 0;
	mesh->y = 0;
	mesh->z = 0;
	mesh->radio = 0;
	mesh->ptIndex = -1;
	mesh->id = meshCount;
	meshCount++;
	return mesh;
}

//	创建根节点网格
//	x, y 为根节点的中心
//	radio 为网格边长的一半
stMesh* createRootMesh( double x, double y, double z, double radio )
{
	stMesh* mesh = createMesh();
	mesh->parent = 0;
	memset( mesh->children, 0, sizeof( stMesh* ) * 8 );
	mesh->level = 0;
	mesh->radio = radio;
	mesh->x = x;
	mesh->y = y;
	mesh->z = z;
	mesh->ptIndex = -1;
	return mesh;
}

//	创建子节点
void createChildrenMesh( stMesh* parent )
{
	if( parent )
	{
		for( int idx = 0; idx < 8; ++idx )
		{
			stMesh* ltM = createMesh();
			ltM->parent = parent;
			ltM->level = parent->level + 1;
			parent->children[ idx ] = ltM;
			ltM->radio = parent->radio * 0.5;
			ltM->ptIndex = -1;
		}
		stMesh* mesh = parent->children[ 0 ];
		mesh->x = parent->x - mesh->radio;
		mesh->y = parent->y - mesh->radio;
		mesh->z = parent->z - mesh->radio;

		mesh = parent->children[ 1 ];
		mesh->x = parent->x + mesh->radio;
		mesh->y = parent->y - mesh->radio;
		mesh->z = parent->z - mesh->radio;

		mesh = parent->children[ 2 ];
		mesh->x = parent->x - mesh->radio;
		mesh->y = parent->y + mesh->radio;
		mesh->z = parent->z - mesh->radio;

		mesh = parent->children[ 3 ];
		mesh->x = parent->x + mesh->radio;
		mesh->y = parent->y + mesh->radio;
		mesh->z = parent->z - mesh->radio;

		mesh = parent->children[ 4 ];
		mesh->x = parent->x - mesh->radio;
		mesh->y = parent->y - mesh->radio;
		mesh->z = parent->z + mesh->radio;

		mesh = parent->children[ 5 ];
		mesh->x = parent->x + mesh->radio;
		mesh->y = parent->y - mesh->radio;
		mesh->z = parent->z + mesh->radio;

		mesh = parent->children[ 6 ];
		mesh->x = parent->x - mesh->radio;
		mesh->y = parent->y + mesh->radio;
		mesh->z = parent->z + mesh->radio;

		mesh = parent->children[ 7 ];
		mesh->x = parent->x + mesh->radio;
		mesh->y = parent->y + mesh->radio;
		mesh->z = parent->z + mesh->radio;

	}
}

stMesh* searchMesh( double x, double y, double z, stMesh* parent )
{
	stMesh* mesh = parent;
	if( fabs( x - mesh->x ) <= mesh->radio &&
		fabs( y - mesh->y ) <= mesh->radio &&
		fabs( z - mesh->z ) <= mesh->radio )
	{
		for( int idx = 0; idx < 8; ++idx )
		{
			if( mesh->children[ idx ] )
			{
				stMesh* m = searchMesh( x, y, z, mesh->children[ idx ] );
				if( m )
					return m;
			}
		}
		return mesh;
	}
	return 0;
}

bool addPtToMesh( int idx, stMesh* parent )
{
	bool result = false;
	stMesh* mesh = searchMesh( x[0][idx], x[1][idx], x[2][idx], parent );
	if( mesh )
	{
		if( mesh->ptIndex == -1 )
		{
			mesh->ptIndex = idx;
			//ptMeshs[ idx ] = mesh;
			result = true;
		}
		else
		{
			int parentId = mesh->ptIndex;
			if( idx != parentId )
			{
				createChildrenMesh( mesh );
				mesh->ptIndex = -1;
				addPtToMesh( parentId, mesh );
				addPtToMesh( idx, mesh );
				result = true;
			}
		}
	}
	else
	{
		printf( "point( %lf, %lf, %lf ) out max range!\n", x[0][idx], x[1][idx], x[2][idx] );
	}
	return result;

}

//	创建网格
//	返回根节点
stMesh* createMeshes()
{
	meshCount = 0;
	stMesh* root = createRootMesh( 2, 2, 2, 3 );
	//ptMeshs.resize( num );
	for( int idx = 0; idx < num; ++idx )
	{
		addPtToMesh( idx, root );
		//if( addPtToMesh( idx, root ) == false )
		//	break;
	}
	return root;
}

void find_pair( stMesh* mesh, int index, double ix, double iy, double iz, double dd )
{
	double mx = mesh->x,
		my = mesh->y,
		mz = mesh->z,
		md = mesh->radio;
	if( fabs( ix - mx ) <= dd + md && fabs( iy - my ) <= dd + md && fabs( iz - mz ) <= dd + md )
	{
		if( mesh->ptIndex != -1 )
		{
			if( mesh->ptIndex > index )
			{
				double x1 = x[0][mesh->ptIndex];
				double y1 = x[1][mesh->ptIndex];
				double z1 = x[2][mesh->ptIndex];
				if( (ix - x1)*(ix - x1) + (iy - y1)*(iy - y1) + (iz - z1)*(iz - z1) < dd * dd )
				{
					pair_i.push_back( index );
					pair_j.push_back( mesh->ptIndex );
				}
			}
		}
		else
		{
			for( int idx = 0; idx < 8; ++idx )
			{
				if( mesh->children[idx] )
				{
					find_pair( mesh->children[ idx ], index, ix, iy, iz, dd );
				}
			}
		}
	}
}

//寻找粒子对(已校验)
// ==================== [OPT-R1] uniform cell list ====================
// 原来的八叉树每步重建 20 万节点并递归下探, 实测 1.16 s/步; 这里换成固定域均匀网格.
// 网格边长 = 搜索半径 3*hsml, 每点只扫 14 个半邻居格(含自身), 每对只出现一次.
static int  g_nxg = 0, g_nyg = 0, g_nzg = 0, g_ncell = 0;
static int* g_cellStart = 0;   // CSR 偏移, 长度 ncell+1
static int* g_cellCur   = 0;   // 填充游标, 长度 ncell
static int* g_sorted    = 0;   // 按 cell 排序后的粒子号, 长度 num
// [OPT-PF] 并行枚举邻对用的两阶段辅助数组
static int* g_cellPairCnt   = 0;   // 每个 cell 的邻对数, 长度 ncell
static int* g_cellPairStart = 0;   // 每个 cell 在 pair 数组里的起点, 长度 ncell+1
static const double g_gridx0 = -0.10, g_gridy0 = -0.10, g_gridz0 = -0.10;
static const double g_gridx1 =  0.70, g_gridy1 =  0.70, g_gridz1 =  1.20;

void init_grid()
{
	const double cs = 3.0*hsml;
	g_nxg = (int)floor((g_gridx1-g_gridx0)/cs)+1;
	g_nyg = (int)floor((g_gridy1-g_gridy0)/cs)+1;
	g_nzg = (int)floor((g_gridz1-g_gridz0)/cs)+1;
	g_ncell = g_nxg*g_nyg*g_nzg;
	g_cellStart = new int[g_ncell+1];
	g_cellCur   = new int[g_ncell];
	g_sorted    = new int[num];
	g_cellPairCnt   = new int[g_ncell];
	g_cellPairStart = new int[g_ncell+1];
	pair_i.reserve(14000000);
	pair_j.reserve(14000000);
}

// [OPT-PF] 枚举某个 cell 的邻对; WRITE=false 时只计数.
// 遍历顺序与串行版完全一致(cell 升序 -> offs 升序 -> (a,b) 升序),
// 因此"先计数、再按预留区间填充"两阶段得到的 pair 顺序与串行逐位相同.
template<bool WRITE>
static int enum_cell(int c, const int (*offs)[3], double R2, int* pi_, int* pj_)
{
	int s0=g_cellStart[c], e0=g_cellStart[c+1];
	if(s0>=e0) return 0;
	int ix=c%g_nxg, iy=(c/g_nxg)%g_nyg, iz=c/(g_nxg*g_nyg);
	int n=0;
	for(int o=0; o<14; ++o)
	{
		int jx=ix+offs[o][0], jy=iy+offs[o][1], jz=iz+offs[o][2];
		if(jx<0||jx>=g_nxg||jy<0||jy>=g_nyg||jz<0||jz>=g_nzg) continue;
		int c2=(jz*g_nyg+jy)*g_nxg+jx;
		int s1=g_cellStart[c2], e1=g_cellStart[c2+1];
		for(int a=s0; a<e0; ++a)
		{
			int pi=g_sorted[a];
			double xi=x[0][pi], yi=x[1][pi], zi=x[2][pi];
			int bb=(o==0)?(a+1):s1;
			int be=(o==0)? e0  :e1;
			for(int b=bb; b<be; ++b)
			{
				int pj=g_sorted[b];
				double d0=xi-x[0][pj], d1=yi-x[1][pj], d2=zi-x[2][pj];
				if(d0*d0+d1*d1+d2*d2<R2)
				{
					int u=pi, v=pj;
					if(u>v){ u=pj; v=pi; }   // 必须保持 i<j: 后面各分支依赖 itype 的粒子编号顺序
					if(WRITE){ pi_[n]=u; pj_[n]=v; }
					++n;
				}
			}
		}
	}
	return n;
}

void direct_find()
{
	const double cs = 3.0*hsml;
	const double R2 = (3.0*hsml)*(3.0*hsml);
	const double inv = 1.0/cs;
	int p, c, ix, iy, iz, acc;
	static const int offs[14][3] = {
		{0,0,0},
		{1,0,0},{-1,1,0},{0,1,0},{1,1,0},
		{-1,-1,1},{0,-1,1},{1,-1,1},{-1,0,1},{0,0,1},{1,0,1},{-1,1,1},{0,1,1},{1,1,1}
	};

	// 1) 计数
	memset(g_cellCur, 0, sizeof(int)*g_ncell);
	for(p=0; p<num; ++p)
	{
		ix=(int)floor((x[0][p]-g_gridx0)*inv); if(ix<0) ix=0; if(ix>=g_nxg) ix=g_nxg-1;
		iy=(int)floor((x[1][p]-g_gridy0)*inv); if(iy<0) iy=0; if(iy>=g_nyg) iy=g_nyg-1;
		iz=(int)floor((x[2][p]-g_gridz0)*inv); if(iz<0) iz=0; if(iz>=g_nzg) iz=g_nzg-1;
		g_cellCur[(iz*g_nyg+iy)*g_nxg+ix]++;
	}
	// 2) 前缀和 + 装桶
	acc = 0;
	for(c=0; c<g_ncell; ++c){ g_cellStart[c]=acc; acc+=g_cellCur[c]; g_cellCur[c]=g_cellStart[c]; }
	g_cellStart[g_ncell]=acc;
	for(p=0; p<num; ++p)
	{
		ix=(int)floor((x[0][p]-g_gridx0)*inv); if(ix<0) ix=0; if(ix>=g_nxg) ix=g_nxg-1;
		iy=(int)floor((x[1][p]-g_gridy0)*inv); if(iy<0) iy=0; if(iy>=g_nyg) iy=g_nyg-1;
		iz=(int)floor((x[2][p]-g_gridz0)*inv); if(iz<0) iz=0; if(iz>=g_nzg) iz=g_nzg-1;
		g_sorted[g_cellCur[(iz*g_nyg+iy)*g_nxg+ix]++] = p;
	}
	// 3) 并行枚举 14 个半邻居格: 先每个 cell 计数, 前缀和定起点, 再并行填充.
	//    pair_i/pair_j 用 capacity 当长度(init_grid 已 reserve 1400 万),
	//    不走 resize, 否则每个时间步都要 memset 掉 89 MB 的 pair 缓冲.
	int total;
	#pragma omp parallel for schedule(dynamic, 32) num_threads(g_omp_nth)
	for(c=0; c<g_ncell; ++c)
	{
		g_cellPairCnt[c] = enum_cell<false>(c, offs, R2, 0, 0);
	}
	total = 0;
	for(c=0; c<g_ncell; ++c){ int t=g_cellPairCnt[c]; g_cellPairStart[c]=total; total+=t; }
	g_cellPairStart[g_ncell]=total;
	if(total > (int)pair_i.capacity())
	{
		pair_i.reserve((size_t)total);
		pair_j.reserve((size_t)total);
	}
	{
		int* piw = pair_i.data();
		int* pjw = pair_j.data();
		#pragma omp parallel for schedule(dynamic, 32) num_threads(g_omp_nth)
		for(c=0; c<g_ncell; ++c)
		{
			int base=g_cellPairStart[c];
			if(g_cellPairCnt[c] > 0)
			{
				enum_cell<true>(c, offs, R2, piw+base, pjw+base);
			}
		}
	}
	niac=total;
}


//寻找粒子对(已校验)
void direct_find1()
{
	/*for(i=0;i<num;i++)
	{
		countiac[i]=0;
	}*/
	pair_i.clear();
	pair_j.clear();
	for(i=0;i<num;i++)
	{
		for(j=i+1;j<num;j++)
		{
			double dr=sqrt((x[0][i]-x[0][j])*(x[0][i]-x[0][j])+(x[1][i]-x[1][j])*(x[1][i]-x[1][j])+(x[2][i]-x[2][j])*(x[2][i]-x[2][j]));
			if(dr<=3.0*hsml) 
			{
				pair_i.push_back(i);
				pair_j.push_back(j);
				//kernel(dr);   //该函数在这里起到何作用了？
				//countiac[i]=countiac[i]+1;
				//countiac[j]=countiac[j]+1;
			}
		}
	}
	niac=(int)pair_i.size();
}
//连续性函数，密度对时间的导数，等于速度的散度(已校验)
void continue_density()
{
	// [OPT-OMP] accumulated quantities are rebuilt by the reduction passes below;
	// only state that is read before any accumulation stays in this serial loop.
	for(i=0;i<num;i++)
	{
		SurFD[i] = SurF[i];
		p[i]=c0*c0*(rho[i]-1000.0);
		if(rho[i]<1000){p[i] = 0;}
	}

	int nth = g_omp_nth;
	if(nth < 1){ nth = 1; }
	if(nth > g_omp_nth){ nth = g_omp_nth; }
	int ii;

	// ---------- pass 1: fluid forces and deformation gradient ----------
	#pragma omp parallel num_threads(nth)
	{
		int tid = omp_get_thread_num();
		double *bb = g_omp_buf + (size_t)tid*(size_t)g_omp_narr*(size_t)num;
		double * __restrict drhodt      = bb + 0*(size_t)num;
		double (* __restrict dvxdt)[num]  = (double (*)[num])(bb + 1*(size_t)num);
		double (* __restrict viscos)[num] = (double (*)[num])(bb + 4*(size_t)num);
		double * __restrict SurF        = bb + 7*(size_t)num;
		double (* __restrict AveV)[num]   = (double (*)[num])(bb + 8*(size_t)num);
		double * __restrict F11    = bb + 11*(size_t)num;
		double * __restrict F12    = bb + 12*(size_t)num;
		double * __restrict F13    = bb + 13*(size_t)num;
		double * __restrict F21    = bb + 14*(size_t)num;
		double * __restrict F22    = bb + 15*(size_t)num;
		double * __restrict F23    = bb + 16*(size_t)num;
		double * __restrict F31    = bb + 17*(size_t)num;
		double * __restrict F32    = bb + 18*(size_t)num;
		double * __restrict F33    = bb + 19*(size_t)num;
		double * __restrict DFdt11 = bb + 20*(size_t)num;
		double * __restrict DFdt12 = bb + 21*(size_t)num;
		double * __restrict DFdt13 = bb + 22*(size_t)num;
		double * __restrict DFdt21 = bb + 23*(size_t)num;
		double * __restrict DFdt22 = bb + 24*(size_t)num;
		double * __restrict DFdt23 = bb + 25*(size_t)num;
		double * __restrict DFdt31 = bb + 26*(size_t)num;
		double * __restrict DFdt32 = bb + 27*(size_t)num;
		double * __restrict DFdt33 = bb + 28*(size_t)num;
		int i, j, k, kk;
		double eta;
		// [OPT-R6] zero only the ranges this thread can write
		for(int tz=0; tz<=10; tz++)
		{
			double * __restrict q = bb + (size_t)tz*(size_t)num;
			for(int m=0; m<ntotal; m++){ q[m] = 0.0; }
		}
		for(int tz=11; tz<=28; tz++)
		{
			double * __restrict q = bb + (size_t)tz*(size_t)num;
			for(int m=ntotal; m<ntotal+SolNum; m++){ q[m] = 0.0; }
		}
		#pragma omp for schedule(dynamic, 16384)
		for(kk=0; kk<niac; kk++)
		{
			k = kk;
		i=pair_i.data()[k];
		j=pair_j.data()[k];
		double dr=sqrt((x[0][i]-x[0][j])*(x[0][i]-x[0][j])+(x[1][i]-x[1][j])*(x[1][i]-x[1][j])+(x[2][i]-x[2][j])*(x[2][i]-x[2][j])+0.0000000000001);
		double e0i = -(x[0][i]-x[0][j])/dr;
		double e1i = -(x[1][i]-x[1][j])/dr;
		double e2i = -(x[2][i]-x[2][j])/dr;

		double e0j = -(x[0][j]-x[0][i])/dr;
		double e1j = -(x[1][j]-x[1][i])/dr;
		double e2j = -(x[2][j]-x[2][i])/dr;
		double AveRHO=0.5*(rho[i]+rho[j]);
		if(fabs(SurFD[i])<2.5){eta = 0.001;}else{eta = 0.00002;}
		if(fabs(SurFD[j])<2.5){eta = 0.001;}else{eta = 0.00002;}
		if(itype[i]>0 && itype[j]>0)
		{
			//i
			double rhoLi = rho[i];              //density L
			double rhoRi = rho[j];				//density R
			double PLi = p[i];                  //PL
			double PRi = p[j];                  //PR
			double ULi = vx[0][i]*e0i + vx[1][i]*e1i + vx[2][i]*e2i;  //UL
			double URi = vx[0][j]*e0i + vx[1][j]*e1i + vx[2][j]*e2i;  //UR

			double UMi = 0.5*(ULi+URi) + 0.5*(PLi-PRi)/(0.5*(rhoLi+rhoRi)*c0);  //U*
 			double PMi = 0.5*(PLi+PRi) + 0.5*(0.5*(rhoLi+rhoRi))*c0*(ULi-URi)*min(3.0*max(ULi-URi,0)/c0,1.0);  //P*
   
			drhodt[i]  = drhodt[i]-2.0*rho[i]*(mass[j]/rho[j])*(ULi-UMi)*dwdr(dr);
			dvxdt[0][i]= dvxdt[0][i]-2.0*mass[j]*(PMi/(rho[i]*rho[j]))*dwdx(x[0][i]-x[0][j],dr);
			dvxdt[1][i]= dvxdt[1][i]-2.0*mass[j]*(PMi/(rho[i]*rho[j]))*dwdx(x[1][i]-x[1][j],dr);
			dvxdt[2][i]= dvxdt[2][i]-2.0*mass[j]*(PMi/(rho[i]*rho[j]))*dwdx(x[2][i]-x[2][j],dr);
			viscos[0][i] = viscos[0][i]+2.0*mass[j]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[0][i]-vx[0][j])/(rho[j]*(dr*dr+0.01*hsml*hsml));
			viscos[1][i] = viscos[1][i]+2.0*mass[j]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[1][i]-vx[1][j])/(rho[j]*(dr*dr+0.01*hsml*hsml));
			viscos[2][i] = viscos[2][i]+2.0*mass[j]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[2][i]-vx[2][j])/(rho[j]*(dr*dr+0.01*hsml*hsml));

			SurF[i] = SurF[i] + ((x[0][i]-x[0][j])*dwdx(x[0][i]-x[0][j],dr)+(x[1][i]-x[1][j])*dwdx(x[1][i]-x[1][j],dr)+(x[2][i]-x[2][j])*dwdx(x[2][i]-x[2][j],dr))*mass[j]/rho[j];
			AveV[0][i]= AveV[0][i]+(mass[j]/AveRHO)*(vx[0][j]-vx[0][i])*kernel(dr);
			AveV[1][i]= AveV[1][i]+(mass[j]/AveRHO)*(vx[1][j]-vx[1][i])*kernel(dr);
			AveV[2][i]= AveV[2][i]+(mass[j]/AveRHO)*(vx[2][j]-vx[2][i])*kernel(dr);

			//j
			double rhoLj = rho[j];
			double rhoRj = rho[i];
			double PLj = p[j];
			double PRj = p[i];
			double ULj = vx[0][j]*e0j + vx[1][j]*e1j + vx[2][j]*e2j;
			double URj = vx[0][i]*e0j + vx[1][i]*e1j + vx[2][i]*e2j;

			double UMj = 0.5*(ULj+URj) + 0.5*(PLj-PRj)/(0.5*(rhoLj+rhoRj)*c0);
 			double PMj = 0.5*(PLj+PRj) + 0.5*(0.5*(rhoLj+rhoRj))*c0*(ULj-URj)*min(3.0*max(ULj-URj,0)/c0,1.0);

			drhodt[j]  = drhodt[j]-2.0*rho[j]*(mass[i]/rho[i])*(ULj-UMj)*dwdr(dr);
			dvxdt[0][j]= dvxdt[0][j]-2.0*mass[i]*(PMj/(rho[i]*rho[j]))*dwdx(x[0][j]-x[0][i],dr);
			dvxdt[1][j]= dvxdt[1][j]-2.0*mass[i]*(PMj/(rho[i]*rho[j]))*dwdx(x[1][j]-x[1][i],dr);
			dvxdt[2][j]= dvxdt[2][j]-2.0*mass[i]*(PMj/(rho[i]*rho[j]))*dwdx(x[2][j]-x[2][i],dr);
			viscos[0][j] = viscos[0][j]+2.0*mass[i]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[0][j]-vx[0][i])/(rho[i]*(dr*dr+0.01*hsml*hsml));
			viscos[1][j] = viscos[1][j]+2.0*mass[i]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[1][j]-vx[1][i])/(rho[i]*(dr*dr+0.01*hsml*hsml));
			viscos[2][j] = viscos[2][j]+2.0*mass[i]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[2][j]-vx[2][i])/(rho[i]*(dr*dr+0.01*hsml*hsml));

			SurF[j] = SurF[j] + ((x[0][i]-x[0][j])*dwdx(x[0][i]-x[0][j],dr)+(x[1][i]-x[1][j])*dwdx(x[1][i]-x[1][j],dr)+(x[2][i]-x[2][j])*dwdx(x[2][i]-x[2][j],dr))*mass[i]/rho[i];
		}
		if(itype[i]>0 && itype[j]<0)
		{
			//i
			double PLi = p[i];
			double PRi = p[i]-rho[i]*9.8*(x[2][j]-x[2][i]);
			double rhoLi = rho[i];
			double rhoRi = 1000.0 + PRi/(c0*c0);//1000*pow(1.0+(7.0*PRi)/(1000*c0*c0),0.14286);  //该式由状态方程反算得出
			double ULi = -vx[0][i]*vecn[0][j]-vx[1][i]*vecn[1][j]-vx[2][i]*vecn[2][j];
			double URi = -ULi;

			double UMi = 0.5*(ULi+URi) + 0.5*(PLi-PRi)/(0.5*(rhoLi+rhoRi)*c0);  //U*
 			double PMi = 0.5*(PLi+PRi) + 0.5*(0.5*(rhoLi+rhoRi))*c0*(ULi-URi)*min(3.0*max(ULi-URi,0)/c0,1.0);  //P*
   
			drhodt[i]  = drhodt[i]-2.0*rho[i]*(mass[j]/rho[j])*(ULi-UMi)*dwdr(dr);
			dvxdt[0][i]= dvxdt[0][i]-2.0*mass[j]*(PMi/(rho[i]*rho[j]))*dwdx(x[0][i]-x[0][j],dr);
			dvxdt[1][i]= dvxdt[1][i]-2.0*mass[j]*(PMi/(rho[i]*rho[j]))*dwdx(x[1][i]-x[1][j],dr);
			dvxdt[2][i]= dvxdt[2][i]-2.0*mass[j]*(PMi/(rho[i]*rho[j]))*dwdx(x[2][i]-x[2][j],dr);
			viscos[0][i] = viscos[0][i]+2.0*mass[j]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[0][i]-vx[0][j])/(rho[j]*(dr*dr+0.01*hsml*hsml));
			viscos[1][i] = viscos[1][i]+2.0*mass[j]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[1][i]-vx[1][j])/(rho[j]*(dr*dr+0.01*hsml*hsml));
			viscos[2][i] = viscos[2][i]+2.0*mass[j]*eta
				*(dwdx(x[0][i]-x[0][j],dr)*(x[0][i]-x[0][j])+dwdx(x[1][i]-x[1][j],dr)*(x[1][i]-x[1][j])+dwdx(x[2][i]-x[2][j],dr)*(x[2][i]-x[2][j]))*(vx[2][i]-vx[2][j])/(rho[j]*(dr*dr+0.01*hsml*hsml));
			SurF[i] = SurF[i] + ((x[0][i]-x[0][j])*dwdx(x[0][i]-x[0][j],dr)+(x[1][i]-x[1][j])*dwdx(x[1][i]-x[1][j],dr)+(x[2][i]-x[2][j])*dwdx(x[2][i]-x[2][j],dr))*mass[j]/rho[j];
		}

		// [OPT-R2] F/DFdt only exist for solid-solid pairs -> skip the rest
		if(itype[i]<-10 && itype[j]<-10)
		{
		double Dr=sqrt((X0[0][i]-X0[0][j])*(X0[0][i]-X0[0][j])+(X0[1][i]-X0[1][j])*(X0[1][i]-X0[1][j])+(X0[2][i]-X0[2][j])*(X0[2][i]-X0[2][j]));
		double gX0=X0[0][i]-X0[0][j];
		double gX1=X0[1][i]-X0[1][j];
		double gX2=X0[2][i]-X0[2][j];
		double g0x=dwdx(gX0,Dr);
		double g0y=dwdx(gX1,Dr);
		double g0z=dwdx(gX2,Dr);
		double dxi0 = x[0][i]-X0[0][i];
		double dxi1 = x[1][i]-X0[1][i];
		double dxi2 = x[2][i]-X0[2][i];
		double dxj0 = x[0][j]-X0[0][j];
		double dxj1 = x[1][j]-X0[1][j];
		double dxj2 = x[2][j]-X0[2][j];

		double Dxi0 = x[0][j]-x[0][i];
		double Dxi1 = x[1][j]-x[1][i];
		double Dxi2 = x[2][j]-x[2][i];
		double Dxj0 = x[0][i]-x[0][j];
		double Dxj1 = x[1][i]-x[1][j];
		double Dxj2 = x[2][i]-x[2][j];

		double vec0i = A11[i]*g0x+A12[i]*g0y+A13[i]*g0z;
		double vec1i = A21[i]*g0x+A22[i]*g0y+A23[i]*g0z;
		double vec2i = A31[i]*g0x+A32[i]*g0y+A33[i]*g0z;
		double vec0j = A11[j]*(-g0x)+A12[j]*(-g0y)+A13[j]*(-g0z);
		double vec1j = A21[j]*(-g0x)+A22[j]*(-g0y)+A23[j]*(-g0z);
		double vec2j = A31[j]*(-g0x)+A32[j]*(-g0y)+A33[j]*(-g0z);

			F11[i]=F11[i]+(mass[j]/rho0[j])*Dxi0*vec0i;
			F12[i]=F12[i]+(mass[j]/rho0[j])*Dxi0*vec1i;
			F13[i]=F13[i]+(mass[j]/rho0[j])*Dxi0*vec2i;
			F21[i]=F21[i]+(mass[j]/rho0[j])*Dxi1*vec0i;
			F22[i]=F22[i]+(mass[j]/rho0[j])*Dxi1*vec1i;
			F23[i]=F23[i]+(mass[j]/rho0[j])*Dxi1*vec2i;
			F31[i]=F31[i]+(mass[j]/rho0[j])*Dxi2*vec0i;
			F32[i]=F32[i]+(mass[j]/rho0[j])*Dxi2*vec1i;
			F33[i]=F33[i]+(mass[j]/rho0[j])*Dxi2*vec2i;

			F11[j]=F11[j]+(mass[i]/rho0[i])*Dxj0*vec0j;
			F12[j]=F12[j]+(mass[i]/rho0[i])*Dxj0*vec1j;
			F13[j]=F13[j]+(mass[i]/rho0[i])*Dxj0*vec2j;
			F21[j]=F21[j]+(mass[i]/rho0[i])*Dxj1*vec0j;
			F22[j]=F22[j]+(mass[i]/rho0[i])*Dxj1*vec1j;
			F23[j]=F23[j]+(mass[i]/rho0[i])*Dxj1*vec2j;
			F31[j]=F31[j]+(mass[i]/rho0[i])*Dxj2*vec0j;
			F32[j]=F32[j]+(mass[i]/rho0[i])*Dxj2*vec1j;
			F33[j]=F33[j]+(mass[i]/rho0[i])*Dxj2*vec2j;

			//摩阻项求解
			DFdt11[i] = DFdt11[i] + (mass[j]/rho0[j])*(vx[0][j]-vx[0][i])*g0x;
			DFdt12[i] = DFdt12[i] + (mass[j]/rho0[j])*(vx[0][j]-vx[0][i])*g0y;
			DFdt13[i] = DFdt13[i] + (mass[j]/rho0[j])*(vx[0][j]-vx[0][i])*g0z;
			DFdt21[i] = DFdt21[i] + (mass[j]/rho0[j])*(vx[1][j]-vx[1][i])*g0x;
			DFdt22[i] = DFdt22[i] + (mass[j]/rho0[j])*(vx[1][j]-vx[1][i])*g0y;
			DFdt23[i] = DFdt23[i] + (mass[j]/rho0[j])*(vx[1][j]-vx[1][i])*g0z;
			DFdt31[i] = DFdt31[i] + (mass[j]/rho0[j])*(vx[2][j]-vx[2][i])*g0x;
			DFdt32[i] = DFdt32[i] + (mass[j]/rho0[j])*(vx[2][j]-vx[2][i])*g0y;
			DFdt33[i] = DFdt33[i] + (mass[j]/rho0[j])*(vx[2][j]-vx[2][i])*g0z;

			DFdt11[j] = DFdt11[j] + (mass[i]/rho0[i])*(vx[0][i]-vx[0][j])*(-g0x);
			DFdt12[j] = DFdt12[j] + (mass[i]/rho0[i])*(vx[0][i]-vx[0][j])*(-g0y);
			DFdt13[j] = DFdt13[j] + (mass[i]/rho0[i])*(vx[0][i]-vx[0][j])*(-g0z);
			DFdt21[j] = DFdt21[j] + (mass[i]/rho0[i])*(vx[1][i]-vx[1][j])*(-g0x);
			DFdt22[j] = DFdt22[j] + (mass[i]/rho0[i])*(vx[1][i]-vx[1][j])*(-g0y);
			DFdt23[j] = DFdt23[j] + (mass[i]/rho0[i])*(vx[1][i]-vx[1][j])*(-g0z);
			DFdt31[j] = DFdt31[j] + (mass[i]/rho0[i])*(vx[2][i]-vx[2][j])*(-g0x);
			DFdt32[j] = DFdt32[j] + (mass[i]/rho0[i])*(vx[2][i]-vx[2][j])*(-g0y);
			DFdt33[j] = DFdt33[j] + (mass[i]/rho0[i])*(vx[2][i]-vx[2][j])*(-g0z);
		}
		}
	}

		// [OPT-R6] restricted reduction over [0, ntotal)
	#pragma omp parallel for schedule(static) num_threads(nth) firstprivate(i)
	for(ii=0; ii<ntotal; ii++)
	{
		i = ii;
		double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0, a7 = 0.0, a8 = 0.0, a9 = 0.0, a10 = 0.0;
		for(int t=0; t<nth; t++)
		{
			const double * __restrict bb = g_omp_buf + (size_t)t*(size_t)g_omp_narr*(size_t)num;
			a0 += bb[0*(size_t)num+i];
			a1 += bb[1*(size_t)num+i];
			a2 += bb[2*(size_t)num+i];
			a3 += bb[3*(size_t)num+i];
			a4 += bb[4*(size_t)num+i];
			a5 += bb[5*(size_t)num+i];
			a6 += bb[6*(size_t)num+i];
			a7 += bb[7*(size_t)num+i];
			a8 += bb[8*(size_t)num+i];
			a9 += bb[9*(size_t)num+i];
			a10 += bb[10*(size_t)num+i];
		}
		drhodt[i] = a0;
		dvxdt[0][i] = a1;
		dvxdt[1][i] = a2;
		dvxdt[2][i] = a3;
		viscos[0][i] = a4;
		viscos[1][i] = a5;
		viscos[2][i] = a6;
		SurF[i] = a7;
		AveV[0][i] = a8;
		AveV[1][i] = a9;
		AveV[2][i] = a10;
	}
	// [OPT-R6] restricted reduction over [ntotal, ntotal+SolNum)
	#pragma omp parallel for schedule(static) num_threads(nth) firstprivate(i)
	for(ii=ntotal; ii<ntotal+SolNum; ii++)
	{
		i = ii;
		double a11 = 0.0, a12 = 0.0, a13 = 0.0, a14 = 0.0, a15 = 0.0, a16 = 0.0, a17 = 0.0, a18 = 0.0, a19 = 0.0, a20 = 0.0, a21 = 0.0, a22 = 0.0, a23 = 0.0, a24 = 0.0, a25 = 0.0, a26 = 0.0, a27 = 0.0, a28 = 0.0;
		for(int t=0; t<nth; t++)
		{
			const double * __restrict bb = g_omp_buf + (size_t)t*(size_t)g_omp_narr*(size_t)num;
			a11 += bb[11*(size_t)num+i];
			a12 += bb[12*(size_t)num+i];
			a13 += bb[13*(size_t)num+i];
			a14 += bb[14*(size_t)num+i];
			a15 += bb[15*(size_t)num+i];
			a16 += bb[16*(size_t)num+i];
			a17 += bb[17*(size_t)num+i];
			a18 += bb[18*(size_t)num+i];
			a19 += bb[19*(size_t)num+i];
			a20 += bb[20*(size_t)num+i];
			a21 += bb[21*(size_t)num+i];
			a22 += bb[22*(size_t)num+i];
			a23 += bb[23*(size_t)num+i];
			a24 += bb[24*(size_t)num+i];
			a25 += bb[25*(size_t)num+i];
			a26 += bb[26*(size_t)num+i];
			a27 += bb[27*(size_t)num+i];
			a28 += bb[28*(size_t)num+i];
		}
		F11[i] = a11;
		F12[i] = a12;
		F13[i] = a13;
		F21[i] = a14;
		F22[i] = a15;
		F23[i] = a16;
		F31[i] = a17;
		F32[i] = a18;
		F33[i] = a19;
		DFdt11[i] = a20;
		DFdt12[i] = a21;
		DFdt13[i] = a22;
		DFdt21[i] = a23;
		DFdt22[i] = a24;
		DFdt23[i] = a25;
		DFdt31[i] = a26;
		DFdt32[i] = a27;
		DFdt33[i] = a28;
	}

	// ---------- solid constitution (needs the reduced F / DFdt) ----------
	for(i=ntotal;i<ntotal+SolNum;i++)//变形梯度张量的修正
	{
		//摩阻项
		DFdtB11[i]=DFdt11[i]*A11[i]+DFdt12[i]*A21[i]+DFdt13[i]*A31[i];
		DFdtB12[i]=DFdt11[i]*A12[i]+DFdt12[i]*A22[i]+DFdt13[i]*A32[i];
		DFdtB13[i]=DFdt11[i]*A13[i]+DFdt12[i]*A23[i]+DFdt13[i]*A33[i];
		DFdtB21[i]=DFdt21[i]*A11[i]+DFdt22[i]*A21[i]+DFdt23[i]*A31[i];
		DFdtB22[i]=DFdt21[i]*A12[i]+DFdt22[i]*A22[i]+DFdt23[i]*A32[i];
		DFdtB23[i]=DFdt21[i]*A13[i]+DFdt22[i]*A23[i]+DFdt23[i]*A33[i];
		DFdtB31[i]=DFdt31[i]*A11[i]+DFdt32[i]*A21[i]+DFdt33[i]*A31[i];
		DFdtB32[i]=DFdt31[i]*A12[i]+DFdt32[i]*A22[i]+DFdt33[i]*A32[i];
		DFdtB33[i]=DFdt31[i]*A13[i]+DFdt32[i]*A23[i]+DFdt33[i]*A33[i];

		double Jacb = F11[i]*F22[i]*F33[i] + F31[i]*F12[i]*F23[i] + F21[i]*F32[i]*F13[i]
			         -F31[i]*F22[i]*F13[i] - F11[i]*F32[i]*F23[i] - F12[i]*F21[i]*F33[i]+0.0;
		rho[i] = rho0[i]/Jacb; 
		if(fabs(Jacb)<0.000001){rho[i]=1000.0;}
		//摩阻项
		double paii=0.25*rho0[i]*1000*hsml;
		SD11[i]=paii*(DFdtB11[i]*F11[i]+DFdtB21[i]*F21[i]+DFdtB31[i]*F31[i] + F11[i]*DFdtB11[i]+F21[i]*DFdtB21[i]+F31[i]*DFdtB31[i]);
		SD12[i]=paii*(DFdtB11[i]*F12[i]+DFdtB21[i]*F22[i]+DFdtB31[i]*F32[i] + F11[i]*DFdtB12[i]+F21[i]*DFdtB22[i]+F31[i]*DFdtB32[i]);
		SD13[i]=paii*(DFdtB11[i]*F13[i]+DFdtB21[i]*F23[i]+DFdtB31[i]*F33[i] + F11[i]*DFdtB13[i]+F21[i]*DFdtB23[i]+F31[i]*DFdtB33[i]);

		SD21[i]=paii*(DFdtB12[i]*F11[i]+DFdtB22[i]*F21[i]+DFdtB32[i]*F31[i] + F12[i]*DFdtB11[i]+F22[i]*DFdtB21[i]+F32[i]*DFdtB31[i]);
		SD22[i]=paii*(DFdtB12[i]*F12[i]+DFdtB22[i]*F22[i]+DFdtB32[i]*F32[i] + F12[i]*DFdtB12[i]+F22[i]*DFdtB22[i]+F32[i]*DFdtB32[i]);
		SD23[i]=paii*(DFdtB12[i]*F13[i]+DFdtB22[i]*F23[i]+DFdtB32[i]*F33[i] + F12[i]*DFdtB13[i]+F22[i]*DFdtB23[i]+F32[i]*DFdtB33[i]);

		SD31[i]=paii*(DFdtB13[i]*F11[i]+DFdtB23[i]*F21[i]+DFdtB33[i]*F31[i] + F13[i]*DFdtB11[i]+F23[i]*DFdtB21[i]+F33[i]*DFdtB31[i]);
		SD32[i]=paii*(DFdtB13[i]*F12[i]+DFdtB23[i]*F22[i]+DFdtB33[i]*F32[i] + F13[i]*DFdtB12[i]+F23[i]*DFdtB22[i]+F33[i]*DFdtB32[i]);
		SD33[i]=paii*(DFdtB13[i]*F13[i]+DFdtB23[i]*F23[i]+DFdtB33[i]*F33[i] + F13[i]*DFdtB13[i]+F23[i]*DFdtB23[i]+F33[i]*DFdtB33[i]);

		//求F的逆
		//double Da=F11[i]*F22[i]*F33[i] + F31[i]*F12[i]*F23[i] + F21[i]*F32[i]*F13[i]
		//	     -F31[i]*F22[i]*F13[i] - F11[i]*F32[i]*F23[i] - F12[i]*F21[i]*F33[i]+0.0;
		//下面使用代数余子式求逆
		Fni11[i]=  (F22[i]*F33[i]-F32[i]*F23[i])/Jacb;
		Fni12[i]= -(F12[i]*F33[i]-F32[i]*F13[i])/Jacb;
		Fni13[i]=  (F12[i]*F23[i]-F22[i]*F13[i])/Jacb;

		Fni21[i]= -(F21[i]*F33[i]-F31[i]*F23[i])/Jacb;
		Fni22[i]=  (F11[i]*F33[i]-F13[i]*F31[i])/Jacb;
		Fni23[i]= -(F11[i]*F23[i]-F21[i]*F13[i])/Jacb;

		Fni31[i]=  (F21[i]*F32[i]-F31[i]*F22[i])/Jacb;
		Fni32[i]= -(F11[i]*F32[i]-F31[i]*F12[i])/Jacb;
		Fni33[i]=  (F11[i]*F22[i]-F21[i]*F12[i])/Jacb;

		//摩阻项
		PD11[i]=F11[i]*SD11[i]+F12[i]*SD21[i]+F13[i]*SD31[i];
		PD12[i]=F11[i]*SD12[i]+F12[i]*SD22[i]+F13[i]*SD32[i];
		PD13[i]=F11[i]*SD13[i]+F12[i]*SD23[i]+F13[i]*SD33[i];

		PD21[i]=F21[i]*SD11[i]+F22[i]*SD21[i]+F23[i]*SD31[i];
		PD22[i]=F21[i]*SD12[i]+F22[i]*SD22[i]+F23[i]*SD32[i];
		PD23[i]=F21[i]*SD13[i]+F22[i]*SD23[i]+F23[i]*SD33[i];

		PD31[i]=F31[i]*SD11[i]+F32[i]*SD21[i]+F33[i]*SD31[i];
		PD32[i]=F31[i]*SD12[i]+F32[i]*SD22[i]+F33[i]*SD32[i];
		PD33[i]=F31[i]*SD13[i]+F32[i]*SD23[i]+F33[i]*SD33[i];

		//求F的逆的转置
		FniT11[i]= Fni11[i];
		FniT12[i]= Fni21[i];
		FniT13[i]= Fni31[i];
		FniT21[i]= Fni12[i];
		FniT22[i]= Fni22[i];
		FniT23[i]= Fni32[i];
		FniT31[i]= Fni13[i];
		FniT32[i]= Fni23[i];
		FniT33[i]= Fni33[i];

		Pia11[i] = muyou*(F11[i]-FniT11[i])+lamd*log10(Jacb)*FniT11[i]+PD11[i];
		Pia12[i] = muyou*(F12[i]-FniT12[i])+lamd*log10(Jacb)*FniT12[i]+PD12[i];
		Pia13[i] = muyou*(F13[i]-FniT13[i])+lamd*log10(Jacb)*FniT13[i]+PD13[i];
		Pia21[i] = muyou*(F21[i]-FniT21[i])+lamd*log10(Jacb)*FniT21[i]+PD21[i];
		Pia22[i] = muyou*(F22[i]-FniT22[i])+lamd*log10(Jacb)*FniT22[i]+PD22[i];
		Pia23[i] = muyou*(F23[i]-FniT23[i])+lamd*log10(Jacb)*FniT23[i]+PD23[i];
		Pia31[i] = muyou*(F31[i]-FniT31[i])+lamd*log10(Jacb)*FniT31[i]+PD31[i];
		Pia32[i] = muyou*(F32[i]-FniT32[i])+lamd*log10(Jacb)*FniT32[i]+PD32[i];
		Pia33[i] = muyou*(F33[i]-FniT33[i])+lamd*log10(Jacb)*FniT33[i]+PD33[i];
	}

	// ---------- pass 2: physical stress and fluid-solid coupling ----------
	#pragma omp parallel num_threads(nth)
	{
		int tid = omp_get_thread_num();
		double *bb = g_omp_buf + (size_t)tid*(size_t)g_omp_narr*(size_t)num;
		double (* __restrict PIJW)[num]   = (double (*)[num])(bb + 29*(size_t)num);
		double (* __restrict SForce)[num] = (double (*)[num])(bb + 32*(size_t)num);
		int i, j, k, kk;
		// [OPT-R6] zero only the ranges this thread can write
		for(int tz=29; tz<=31; tz++)
		{
			double * __restrict q = bb + (size_t)tz*(size_t)num;
			for(int m=ntotal; m<ntotal+SolNum; m++){ q[m] = 0.0; }
		}
		for(int tz=32; tz<=34; tz++)
		{
			double * __restrict q = bb + (size_t)tz*(size_t)num;
			for(int m=0; m<ntotal+SolNum; m++){ q[m] = 0.0; }
		}
		#pragma omp for schedule(dynamic, 16384)
		for(kk=0; kk<niac; kk++)
		{
			k = kk;
		i=pair_i.data()[k];
		j=pair_j.data()[k];
		// [OPT-R2] this loop only serves pairs containing a solid particle
		if(!(itype[i]<-10 || itype[j]<-10)) continue;
		double dr=sqrt((x[0][i]-x[0][j])*(x[0][i]-x[0][j])+(x[1][i]-x[1][j])*(x[1][i]-x[1][j])+(x[2][i]-x[2][j])*(x[2][i]-x[2][j]));
		double Dr=sqrt((X0[0][i]-X0[0][j])*(X0[0][i]-X0[0][j])+(X0[1][i]-X0[1][j])*(X0[1][i]-X0[1][j])+(X0[2][i]-X0[2][j])*(X0[2][i]-X0[2][j]));
		double gX0=X0[0][i]-X0[0][j];
		double gX1=X0[1][i]-X0[1][j];
		double gX2=X0[2][i]-X0[2][j];
		double g0x=dwdx(gX0,Dr);
		double g0y=dwdx(gX1,Dr);
		double g0z=dwdx(gX2,Dr);

		double vec0i = A11[i]*g0x+A12[i]*g0y+A13[i]*g0z;
		double vec1i = A21[i]*g0x+A22[i]*g0y+A23[i]*g0z;
		double vec2i = A31[i]*g0x+A32[i]*g0y+A33[i]*g0z;
		double vec0j = A11[j]*(-g0x)+A12[j]*(-g0y)+A13[j]*(-g0z);
		double vec1j = A21[j]*(-g0x)+A22[j]*(-g0y)+A23[j]*(-g0z);
		double vec2j = A31[j]*(-g0x)+A32[j]*(-g0y)+A33[j]*(-g0z);

		double Pi0 = Pia11[i]*vec0i+Pia12[i]*vec1i+Pia13[i]*vec2i;
		double Pi1 = Pia21[i]*vec0i+Pia22[i]*vec1i+Pia23[i]*vec2i;
		double Pi2 = Pia31[i]*vec0i+Pia32[i]*vec1i+Pia33[i]*vec2i;

		double Pj0 = Pia11[j]*vec0j+Pia12[j]*vec1j+Pia13[j]*vec2j;
		double Pj1 = Pia21[j]*vec0j+Pia22[j]*vec1j+Pia23[j]*vec2j;
		double Pj2 = Pia31[j]*vec0j+Pia32[j]*vec1j+Pia33[j]*vec2j;
		if(itype[i]<-10 && itype[j]<-10)
		{
			PIJW[0][i]=PIJW[0][i]+(mass[j]/(rho[i]*rho[j]))*(Pi0-Pj0);
			PIJW[1][i]=PIJW[1][i]+(mass[j]/(rho[i]*rho[j]))*(Pi1-Pj1);
			PIJW[2][i]=PIJW[2][i]+(mass[j]/(rho[i]*rho[j]))*(Pi2-Pj2);

			PIJW[0][j]=PIJW[0][j]+(mass[i]/(rho[i]*rho[j]))*(Pj0-Pi0);
			PIJW[1][j]=PIJW[1][j]+(mass[i]/(rho[i]*rho[j]))*(Pj1-Pi1);
			PIJW[2][j]=PIJW[2][j]+(mass[i]/(rho[i]*rho[j]))*(Pj2-Pi2);
		}
		if(itype[i]>0 && itype[j]<-10)
		{
			double PRii = p[i]+rho[i]*max(0,(0-PIJW[0][i])*vecn[0][j]+(0-PIJW[1][i])*vecn[1][j]+(0-PIJW[2][i])*vecn[2][j])
				*((x[0][i]-x[0][j])*vecn[0][j]+(x[1][i]-x[1][j])*vecn[1][j]+(x[2][i]-x[2][j])*vecn[2][j])
				*((x[0][i]-x[0][j])*vecn[0][j]+(x[1][i]-x[1][j])*vecn[1][j]+(x[2][i]-x[2][j])*vecn[2][j]);

			SForce[0][i] = SForce[0][i] - 2.0*(mass[j]/(rho[i]*rho[j]))*dwdx(x[0][i]-x[0][j],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[1][i] = SForce[1][i] - 2.0*(mass[j]/(rho[i]*rho[j]))*dwdx(x[1][i]-x[1][j],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[2][i] = SForce[2][i] - 2.0*(mass[j]/(rho[i]*rho[j]))*dwdx(x[2][i]-x[2][j],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[0][j] = SForce[0][j] - 2.0*(mass[i]/(rho[i]*rho[j]))*dwdx(x[0][j]-x[0][i],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[1][j] = SForce[1][j] - 2.0*(mass[i]/(rho[i]*rho[j]))*dwdx(x[1][j]-x[1][i],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[2][j] = SForce[2][j] - 2.0*(mass[i]/(rho[i]*rho[j]))*dwdx(x[2][j]-x[2][i],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
		}
		}
	}

		// [OPT-R6] restricted reduction over [0, ntotal+SolNum)
	#pragma omp parallel for schedule(static) num_threads(nth) firstprivate(i)
	for(ii=0; ii<ntotal+SolNum; ii++)
	{
		i = ii;
		double a3 = 0.0, a4 = 0.0, a5 = 0.0;
		for(int t=0; t<nth; t++)
		{
			const double * __restrict bb = g_omp_buf + (size_t)t*(size_t)g_omp_narr*(size_t)num;
			a3 += bb[32*(size_t)num+i];
			a4 += bb[33*(size_t)num+i];
			a5 += bb[34*(size_t)num+i];
		}
		SForce[0][i] = a3;
		SForce[1][i] = a4;
		SForce[2][i] = a5;
	}
	// [OPT-R6] restricted reduction over [ntotal, ntotal+SolNum)
	#pragma omp parallel for schedule(static) num_threads(nth) firstprivate(i)
	for(ii=ntotal; ii<ntotal+SolNum; ii++)
	{
		i = ii;
		double a0 = 0.0, a1 = 0.0, a2 = 0.0;
		for(int t=0; t<nth; t++)
		{
			const double * __restrict bb = g_omp_buf + (size_t)t*(size_t)g_omp_narr*(size_t)num;
			a0 += bb[29*(size_t)num+i];
			a1 += bb[30*(size_t)num+i];
			a2 += bb[31*(size_t)num+i];
		}
		PIJW[0][i] = a0;
		PIJW[1][i] = a1;
		PIJW[2][i] = a2;
	}
}

void GraCorrec()
{
	for(i=0;i<num;i++)
	{
		a11[i]=0.0; a12[i]=0.0; a13[i]=0.0;
		a21[i]=0.0;	a22[i]=0.0; a23[i]=0.0;
		a31[i]=0.0;	a32[i]=0.0; a33[i]=0.0;
	}
	for(k=0;k<niac;k++)
	{
		i=pair_i.data()[k];
		j=pair_j.data()[k];
		double Dr=sqrt((X0[0][i]-X0[0][j])*(X0[0][i]-X0[0][j])+(X0[1][i]-X0[1][j])*(X0[1][i]-X0[1][j])+(X0[2][i]-X0[2][j])*(X0[2][i]-X0[2][j]));
		if(itype[i]<-10 && itype[j]<-10)
		{
			a11[i]=a11[i]+(mass[j]/rho0[j])*(X0[0][j]-X0[0][i])*dwdx(X0[0][i]-X0[0][j],Dr);
			a12[i]=a12[i]+(mass[j]/rho0[j])*(X0[0][j]-X0[0][i])*dwdx(X0[1][i]-X0[1][j],Dr);
			a13[i]=a13[i]+(mass[j]/rho0[j])*(X0[0][j]-X0[0][i])*dwdx(X0[2][i]-X0[2][j],Dr);
			a21[i]=a21[i]+(mass[j]/rho0[j])*(X0[1][j]-X0[1][i])*dwdx(X0[0][i]-X0[0][j],Dr);
			a22[i]=a22[i]+(mass[j]/rho0[j])*(X0[1][j]-X0[1][i])*dwdx(X0[1][i]-X0[1][j],Dr);
			a23[i]=a23[i]+(mass[j]/rho0[j])*(X0[1][j]-X0[1][i])*dwdx(X0[2][i]-X0[2][j],Dr);
			a31[i]=a31[i]+(mass[j]/rho0[j])*(X0[2][j]-X0[2][i])*dwdx(X0[0][i]-X0[0][j],Dr);
			a32[i]=a32[i]+(mass[j]/rho0[j])*(X0[2][j]-X0[2][i])*dwdx(X0[1][i]-X0[1][j],Dr);
			a33[i]=a33[i]+(mass[j]/rho0[j])*(X0[2][j]-X0[2][i])*dwdx(X0[2][i]-X0[2][j],Dr);

			a11[j]=a11[j]+(mass[i]/rho0[i])*(X0[0][i]-X0[0][j])*dwdx(X0[0][j]-X0[0][i],Dr);
			a12[j]=a12[j]+(mass[i]/rho0[i])*(X0[0][i]-X0[0][j])*dwdx(X0[1][j]-X0[1][i],Dr);
			a13[j]=a13[j]+(mass[i]/rho0[i])*(X0[0][i]-X0[0][j])*dwdx(X0[2][j]-X0[2][i],Dr);
			a21[j]=a21[j]+(mass[i]/rho0[i])*(X0[1][i]-X0[1][j])*dwdx(X0[0][j]-X0[0][i],Dr);
			a22[j]=a22[j]+(mass[i]/rho0[i])*(X0[1][i]-X0[1][j])*dwdx(X0[1][j]-X0[1][i],Dr);
			a23[j]=a23[j]+(mass[i]/rho0[i])*(X0[1][i]-X0[1][j])*dwdx(X0[2][j]-X0[2][i],Dr);
			a31[j]=a31[j]+(mass[i]/rho0[i])*(X0[2][i]-X0[2][j])*dwdx(X0[0][j]-X0[0][i],Dr);
			a32[j]=a32[j]+(mass[i]/rho0[i])*(X0[2][i]-X0[2][j])*dwdx(X0[1][j]-X0[1][i],Dr);
			a33[j]=a33[j]+(mass[i]/rho0[i])*(X0[2][i]-X0[2][j])*dwdx(X0[2][j]-X0[2][i],Dr);
		}
	}
	for(i=ntotal;i<ntotal+SolNum;i++)
	{
		double Da=a11[i]*a22[i]*a33[i] + a31[i]*a12[i]*a23[i] + a21[i]*a32[i]*a13[i]
			     -a31[i]*a22[i]*a13[i] - a11[i]*a32[i]*a23[i] - a12[i]*a21[i]*a33[i]+0.0;
		//下面使用代数余子式求逆
		A11[i]=  (a22[i]*a33[i]-a32[i]*a23[i])/Da;
		A12[i]= -(a12[i]*a33[i]-a32[i]*a13[i])/Da;
		A13[i]=  (a12[i]*a23[i]-a22[i]*a13[i])/Da;

		A21[i]= -(a21[i]*a33[i]-a31[i]*a23[i])/Da;
		A22[i]=  (a11[i]*a33[i]-a13[i]*a31[i])/Da;
		A23[i]= -(a11[i]*a23[i]-a21[i]*a13[i])/Da;

		A31[i]=  (a21[i]*a32[i]-a31[i]*a22[i])/Da;
		A32[i]= -(a11[i]*a32[i]-a31[i]*a12[i])/Da;
		A33[i]=  (a11[i]*a22[i]-a21[i]*a12[i])/Da;
		if(fabs(Da)<0.000001)
		{A11[i]=1.0; A12[i]=0.0; A13[i]=0.0;
		 A21[i]=0.0; A22[i]=1.0; A23[i]=0.0;
 		 A31[i]=0.0; A32[i]=0.0; A33[i]=1.0;}/**/
	}
}
void GLStrain()
{
	for(i=0;i<num;i++)
	{
		F11[i]=0.0; F12[i]=0.0; F13[i]=0.0;
		F21[i]=0.0; F22[i]=0.0; F23[i]=0.0;
		F31[i]=0.0; F32[i]=0.0; F33[i]=0.0;
		PIJW[0][i]=0.0;
		PIJW[1][i]=0.0;
		PIJW[2][i]=0.0;
		SForce[0][i]=0.0;
		SForce[1][i]=0.0;
		SForce[2][i]=0.0;

		//摩阻项求解
		DFdt11[i]=0.0; DFdt12[i] = 0.0; DFdt13[i] = 0.0;
		DFdt21[i]=0.0; DFdt22[i] = 0.0; DFdt23[i] = 0.0;
		DFdt31[i]=0.0; DFdt32[i] = 0.0; DFdt33[i] = 0.0;
	}
	for(k=0;k<niac;k++)//变形梯度张量
	{
		i=pair_i.data()[k];
		j=pair_j.data()[k];
		double dr=sqrt((x[0][i]-x[0][j])*(x[0][i]-x[0][j])+(x[1][i]-x[1][j])*(x[1][i]-x[1][j])+(x[2][i]-x[2][j])*(x[2][i]-x[2][j]));
		double Dr=sqrt((X0[0][i]-X0[0][j])*(X0[0][i]-X0[0][j])+(X0[1][i]-X0[1][j])*(X0[1][i]-X0[1][j])+(X0[2][i]-X0[2][j])*(X0[2][i]-X0[2][j]));
		double dxi0 = x[0][i]-X0[0][i];
		double dxi1 = x[1][i]-X0[1][i];
		double dxi2 = x[2][i]-X0[2][i];
		double dxj0 = x[0][j]-X0[0][j];
		double dxj1 = x[1][j]-X0[1][j];
		double dxj2 = x[2][j]-X0[2][j];

		double Dxi0 = x[0][j]-x[0][i];
		double Dxi1 = x[1][j]-x[1][i];
		double Dxi2 = x[2][j]-x[2][i];
		double Dxj0 = x[0][i]-x[0][j];
		double Dxj1 = x[1][i]-x[1][j];
		double Dxj2 = x[2][i]-x[2][j];

		double vec0i = A11[i]*dwdx(X0[0][i]-X0[0][j],Dr)+A12[i]*dwdx(X0[1][i]-X0[1][j],Dr)+A13[i]*dwdx(X0[2][i]-X0[2][j],Dr);
		double vec1i = A21[i]*dwdx(X0[0][i]-X0[0][j],Dr)+A22[i]*dwdx(X0[1][i]-X0[1][j],Dr)+A23[i]*dwdx(X0[2][i]-X0[2][j],Dr);
		double vec2i = A31[i]*dwdx(X0[0][i]-X0[0][j],Dr)+A32[i]*dwdx(X0[1][i]-X0[1][j],Dr)+A33[i]*dwdx(X0[2][i]-X0[2][j],Dr);
		double vec0j = A11[j]*dwdx(X0[0][j]-X0[0][i],Dr)+A12[j]*dwdx(X0[1][j]-X0[1][i],Dr)+A13[j]*dwdx(X0[2][j]-X0[2][i],Dr);
		double vec1j = A21[j]*dwdx(X0[0][j]-X0[0][i],Dr)+A22[j]*dwdx(X0[1][j]-X0[1][i],Dr)+A23[j]*dwdx(X0[2][j]-X0[2][i],Dr);
		double vec2j = A31[j]*dwdx(X0[0][j]-X0[0][i],Dr)+A32[j]*dwdx(X0[1][j]-X0[1][i],Dr)+A33[j]*dwdx(X0[2][j]-X0[2][i],Dr);

		if(itype[i]<-10 && itype[j]<-10)
		{
			F11[i]=F11[i]+(mass[j]/rho0[j])*Dxi0*vec0i;
			F12[i]=F12[i]+(mass[j]/rho0[j])*Dxi0*vec1i;
			F13[i]=F13[i]+(mass[j]/rho0[j])*Dxi0*vec2i;
			F21[i]=F21[i]+(mass[j]/rho0[j])*Dxi1*vec0i;
			F22[i]=F22[i]+(mass[j]/rho0[j])*Dxi1*vec1i;
			F23[i]=F23[i]+(mass[j]/rho0[j])*Dxi1*vec2i;
			F31[i]=F31[i]+(mass[j]/rho0[j])*Dxi2*vec0i;
			F32[i]=F32[i]+(mass[j]/rho0[j])*Dxi2*vec1i;
			F33[i]=F33[i]+(mass[j]/rho0[j])*Dxi2*vec2i;

			F11[j]=F11[j]+(mass[i]/rho0[i])*Dxj0*vec0j;
			F12[j]=F12[j]+(mass[i]/rho0[i])*Dxj0*vec1j;
			F13[j]=F13[j]+(mass[i]/rho0[i])*Dxj0*vec2j;
			F21[j]=F21[j]+(mass[i]/rho0[i])*Dxj1*vec0j;
			F22[j]=F22[j]+(mass[i]/rho0[i])*Dxj1*vec1j;
			F23[j]=F23[j]+(mass[i]/rho0[i])*Dxj1*vec2j;
			F31[j]=F31[j]+(mass[i]/rho0[i])*Dxj2*vec0j;
			F32[j]=F32[j]+(mass[i]/rho0[i])*Dxj2*vec1j;
			F33[j]=F33[j]+(mass[i]/rho0[i])*Dxj2*vec2j;

			//摩阻项求解
			DFdt11[i] = DFdt11[i] + (mass[j]/rho0[j])*(vx[0][j]-vx[0][i])*dwdx(X0[0][i]-X0[0][j],Dr);
			DFdt12[i] = DFdt12[i] + (mass[j]/rho0[j])*(vx[0][j]-vx[0][i])*dwdx(X0[1][i]-X0[1][j],Dr);
			DFdt13[i] = DFdt13[i] + (mass[j]/rho0[j])*(vx[0][j]-vx[0][i])*dwdx(X0[2][i]-X0[2][j],Dr);
			DFdt21[i] = DFdt21[i] + (mass[j]/rho0[j])*(vx[1][j]-vx[1][i])*dwdx(X0[0][i]-X0[0][j],Dr);
			DFdt22[i] = DFdt22[i] + (mass[j]/rho0[j])*(vx[1][j]-vx[1][i])*dwdx(X0[1][i]-X0[1][j],Dr);
			DFdt23[i] = DFdt23[i] + (mass[j]/rho0[j])*(vx[1][j]-vx[1][i])*dwdx(X0[2][i]-X0[2][j],Dr);
			DFdt31[i] = DFdt31[i] + (mass[j]/rho0[j])*(vx[2][j]-vx[2][i])*dwdx(X0[0][i]-X0[0][j],Dr);
			DFdt32[i] = DFdt32[i] + (mass[j]/rho0[j])*(vx[2][j]-vx[2][i])*dwdx(X0[1][i]-X0[1][j],Dr);
			DFdt33[i] = DFdt33[i] + (mass[j]/rho0[j])*(vx[2][j]-vx[2][i])*dwdx(X0[2][i]-X0[2][j],Dr);

			DFdt11[j] = DFdt11[j] + (mass[i]/rho0[i])*(vx[0][i]-vx[0][j])*dwdx(X0[0][j]-X0[0][i],Dr);
			DFdt12[j] = DFdt12[j] + (mass[i]/rho0[i])*(vx[0][i]-vx[0][j])*dwdx(X0[1][j]-X0[1][i],Dr);
			DFdt13[j] = DFdt13[j] + (mass[i]/rho0[i])*(vx[0][i]-vx[0][j])*dwdx(X0[2][j]-X0[2][i],Dr);
			DFdt21[j] = DFdt21[j] + (mass[i]/rho0[i])*(vx[1][i]-vx[1][j])*dwdx(X0[0][j]-X0[0][i],Dr);
			DFdt22[j] = DFdt22[j] + (mass[i]/rho0[i])*(vx[1][i]-vx[1][j])*dwdx(X0[1][j]-X0[1][i],Dr);
			DFdt23[j] = DFdt23[j] + (mass[i]/rho0[i])*(vx[1][i]-vx[1][j])*dwdx(X0[2][j]-X0[2][i],Dr);
			DFdt31[j] = DFdt31[j] + (mass[i]/rho0[i])*(vx[2][i]-vx[2][j])*dwdx(X0[0][j]-X0[0][i],Dr);
			DFdt32[j] = DFdt32[j] + (mass[i]/rho0[i])*(vx[2][i]-vx[2][j])*dwdx(X0[1][j]-X0[1][i],Dr);
			DFdt33[j] = DFdt33[j] + (mass[i]/rho0[i])*(vx[2][i]-vx[2][j])*dwdx(X0[2][j]-X0[2][i],Dr);
		}
	}
	for(i=ntotal;i<ntotal+SolNum;i++)//变形梯度张量的修正
	{
		//摩阻项
		DFdtB11[i]=DFdt11[i]*A11[i]+DFdt12[i]*A21[i]+DFdt13[i]*A31[i];
		DFdtB12[i]=DFdt11[i]*A12[i]+DFdt12[i]*A22[i]+DFdt13[i]*A32[i];
		DFdtB13[i]=DFdt11[i]*A13[i]+DFdt12[i]*A23[i]+DFdt13[i]*A33[i];
		DFdtB21[i]=DFdt21[i]*A11[i]+DFdt22[i]*A21[i]+DFdt23[i]*A31[i];
		DFdtB22[i]=DFdt21[i]*A12[i]+DFdt22[i]*A22[i]+DFdt23[i]*A32[i];
		DFdtB23[i]=DFdt21[i]*A13[i]+DFdt22[i]*A23[i]+DFdt23[i]*A33[i];
		DFdtB31[i]=DFdt31[i]*A11[i]+DFdt32[i]*A21[i]+DFdt33[i]*A31[i];
		DFdtB32[i]=DFdt31[i]*A12[i]+DFdt32[i]*A22[i]+DFdt33[i]*A32[i];
		DFdtB33[i]=DFdt31[i]*A13[i]+DFdt32[i]*A23[i]+DFdt33[i]*A33[i];

		double Jacb = F11[i]*F22[i]*F33[i] + F31[i]*F12[i]*F23[i] + F21[i]*F32[i]*F13[i]
			         -F31[i]*F22[i]*F13[i] - F11[i]*F32[i]*F23[i] - F12[i]*F21[i]*F33[i]+0.0;
		rho[i] = rho0[i]/Jacb; 
		if(fabs(Jacb)<0.000001){rho[i]=1000.0;}
		//摩阻项
		double paii=0.25*rho0[i]*1000*hsml;
		SD11[i]=paii*(DFdtB11[i]*F11[i]+DFdtB21[i]*F21[i]+DFdtB31[i]*F31[i] + F11[i]*DFdtB11[i]+F21[i]*DFdtB21[i]+F31[i]*DFdtB31[i]);
		SD12[i]=paii*(DFdtB11[i]*F12[i]+DFdtB21[i]*F22[i]+DFdtB31[i]*F32[i] + F11[i]*DFdtB12[i]+F21[i]*DFdtB22[i]+F31[i]*DFdtB32[i]);
		SD13[i]=paii*(DFdtB11[i]*F13[i]+DFdtB21[i]*F23[i]+DFdtB31[i]*F33[i] + F11[i]*DFdtB13[i]+F21[i]*DFdtB23[i]+F31[i]*DFdtB33[i]);

		SD21[i]=paii*(DFdtB12[i]*F11[i]+DFdtB22[i]*F21[i]+DFdtB32[i]*F31[i] + F12[i]*DFdtB11[i]+F22[i]*DFdtB21[i]+F32[i]*DFdtB31[i]);
		SD22[i]=paii*(DFdtB12[i]*F12[i]+DFdtB22[i]*F22[i]+DFdtB32[i]*F32[i] + F12[i]*DFdtB12[i]+F22[i]*DFdtB22[i]+F32[i]*DFdtB32[i]);
		SD23[i]=paii*(DFdtB12[i]*F13[i]+DFdtB22[i]*F23[i]+DFdtB32[i]*F33[i] + F12[i]*DFdtB13[i]+F22[i]*DFdtB23[i]+F32[i]*DFdtB33[i]);

		SD31[i]=paii*(DFdtB13[i]*F11[i]+DFdtB23[i]*F21[i]+DFdtB33[i]*F31[i] + F13[i]*DFdtB11[i]+F23[i]*DFdtB21[i]+F33[i]*DFdtB31[i]);
		SD32[i]=paii*(DFdtB13[i]*F12[i]+DFdtB23[i]*F22[i]+DFdtB33[i]*F32[i] + F13[i]*DFdtB12[i]+F23[i]*DFdtB22[i]+F33[i]*DFdtB32[i]);
		SD33[i]=paii*(DFdtB13[i]*F13[i]+DFdtB23[i]*F23[i]+DFdtB33[i]*F33[i] + F13[i]*DFdtB13[i]+F23[i]*DFdtB23[i]+F33[i]*DFdtB33[i]);

		//求F的逆
		//double Da=F11[i]*F22[i]*F33[i] + F31[i]*F12[i]*F23[i] + F21[i]*F32[i]*F13[i]
		//	     -F31[i]*F22[i]*F13[i] - F11[i]*F32[i]*F23[i] - F12[i]*F21[i]*F33[i]+0.0;
		//下面使用代数余子式求逆
		Fni11[i]=  (F22[i]*F33[i]-F32[i]*F23[i])/Jacb;
		Fni12[i]= -(F12[i]*F33[i]-F32[i]*F13[i])/Jacb;
		Fni13[i]=  (F12[i]*F23[i]-F22[i]*F13[i])/Jacb;

		Fni21[i]= -(F21[i]*F33[i]-F31[i]*F23[i])/Jacb;
		Fni22[i]=  (F11[i]*F33[i]-F13[i]*F31[i])/Jacb;
		Fni23[i]= -(F11[i]*F23[i]-F21[i]*F13[i])/Jacb;

		Fni31[i]=  (F21[i]*F32[i]-F31[i]*F22[i])/Jacb;
		Fni32[i]= -(F11[i]*F32[i]-F31[i]*F12[i])/Jacb;
		Fni33[i]=  (F11[i]*F22[i]-F21[i]*F12[i])/Jacb;

		//摩阻项
		PD11[i]=F11[i]*SD11[i]+F12[i]*SD21[i]+F13[i]*SD31[i];
		PD12[i]=F11[i]*SD12[i]+F12[i]*SD22[i]+F13[i]*SD32[i];
		PD13[i]=F11[i]*SD13[i]+F12[i]*SD23[i]+F13[i]*SD33[i];

		PD21[i]=F21[i]*SD11[i]+F22[i]*SD21[i]+F23[i]*SD31[i];
		PD22[i]=F21[i]*SD12[i]+F22[i]*SD22[i]+F23[i]*SD32[i];
		PD23[i]=F21[i]*SD13[i]+F22[i]*SD23[i]+F23[i]*SD33[i];

		PD31[i]=F31[i]*SD11[i]+F32[i]*SD21[i]+F33[i]*SD31[i];
		PD32[i]=F31[i]*SD12[i]+F32[i]*SD22[i]+F33[i]*SD32[i];
		PD33[i]=F31[i]*SD13[i]+F32[i]*SD23[i]+F33[i]*SD33[i];

		//求F的逆的转置
		FniT11[i]= Fni11[i];
		FniT12[i]= Fni21[i];
		FniT13[i]= Fni31[i];
		FniT21[i]= Fni12[i];
		FniT22[i]= Fni22[i];
		FniT23[i]= Fni32[i];
		FniT31[i]= Fni13[i];
		FniT32[i]= Fni23[i];
		FniT33[i]= Fni33[i];

		Pia11[i] = muyou*(F11[i]-FniT11[i])+lamd*log10(Jacb)*FniT11[i]+PD11[i];
		Pia12[i] = muyou*(F12[i]-FniT12[i])+lamd*log10(Jacb)*FniT12[i]+PD12[i];
		Pia13[i] = muyou*(F13[i]-FniT13[i])+lamd*log10(Jacb)*FniT13[i]+PD13[i];
		Pia21[i] = muyou*(F21[i]-FniT21[i])+lamd*log10(Jacb)*FniT21[i]+PD21[i];
		Pia22[i] = muyou*(F22[i]-FniT22[i])+lamd*log10(Jacb)*FniT22[i]+PD22[i];
		Pia23[i] = muyou*(F23[i]-FniT23[i])+lamd*log10(Jacb)*FniT23[i]+PD23[i];
		Pia31[i] = muyou*(F31[i]-FniT31[i])+lamd*log10(Jacb)*FniT31[i]+PD31[i];
		Pia32[i] = muyou*(F32[i]-FniT32[i])+lamd*log10(Jacb)*FniT32[i]+PD32[i];
		Pia33[i] = muyou*(F33[i]-FniT33[i])+lamd*log10(Jacb)*FniT33[i]+PD33[i];
	}
	for(k=0;k<niac;k++)//变形梯度张量
	{
		i=pair_i.data()[k];
		j=pair_j.data()[k];
		double dr=sqrt((x[0][i]-x[0][j])*(x[0][i]-x[0][j])+(x[1][i]-x[1][j])*(x[1][i]-x[1][j])+(x[2][i]-x[2][j])*(x[2][i]-x[2][j]));
		double Dr=sqrt((X0[0][i]-X0[0][j])*(X0[0][i]-X0[0][j])+(X0[1][i]-X0[1][j])*(X0[1][i]-X0[1][j])+(X0[2][i]-X0[2][j])*(X0[2][i]-X0[2][j]));

		double vec0i = A11[i]*dwdx(X0[0][i]-X0[0][j],Dr)+A12[i]*dwdx(X0[1][i]-X0[1][j],Dr)+A13[i]*dwdx(X0[2][i]-X0[2][j],Dr);
		double vec1i = A21[i]*dwdx(X0[0][i]-X0[0][j],Dr)+A22[i]*dwdx(X0[1][i]-X0[1][j],Dr)+A23[i]*dwdx(X0[2][i]-X0[2][j],Dr);
		double vec2i = A31[i]*dwdx(X0[0][i]-X0[0][j],Dr)+A32[i]*dwdx(X0[1][i]-X0[1][j],Dr)+A33[i]*dwdx(X0[2][i]-X0[2][j],Dr);
		double vec0j = A11[j]*dwdx(X0[0][j]-X0[0][i],Dr)+A12[j]*dwdx(X0[1][j]-X0[1][i],Dr)+A13[j]*dwdx(X0[2][j]-X0[2][i],Dr);
		double vec1j = A21[j]*dwdx(X0[0][j]-X0[0][i],Dr)+A22[j]*dwdx(X0[1][j]-X0[1][i],Dr)+A23[j]*dwdx(X0[2][j]-X0[2][i],Dr);
		double vec2j = A31[j]*dwdx(X0[0][j]-X0[0][i],Dr)+A32[j]*dwdx(X0[1][j]-X0[1][i],Dr)+A33[j]*dwdx(X0[2][j]-X0[2][i],Dr);

		double Pi0 = Pia11[i]*vec0i+Pia12[i]*vec1i+Pia13[i]*vec2i;
		double Pi1 = Pia21[i]*vec0i+Pia22[i]*vec1i+Pia23[i]*vec2i;
		double Pi2 = Pia31[i]*vec0i+Pia32[i]*vec1i+Pia33[i]*vec2i;

		double Pj0 = Pia11[j]*vec0j+Pia12[j]*vec1j+Pia13[j]*vec2j;
		double Pj1 = Pia21[j]*vec0j+Pia22[j]*vec1j+Pia23[j]*vec2j;
		double Pj2 = Pia31[j]*vec0j+Pia32[j]*vec1j+Pia33[j]*vec2j;
		if(itype[i]<-10 && itype[j]<-10)
		{
			PIJW[0][i]=PIJW[0][i]+(mass[j]/(rho[i]*rho[j]))*(Pi0-Pj0);
			PIJW[1][i]=PIJW[1][i]+(mass[j]/(rho[i]*rho[j]))*(Pi1-Pj1);
			PIJW[2][i]=PIJW[2][i]+(mass[j]/(rho[i]*rho[j]))*(Pi2-Pj2);

			PIJW[0][j]=PIJW[0][j]+(mass[i]/(rho[i]*rho[j]))*(Pj0-Pi0);
			PIJW[1][j]=PIJW[1][j]+(mass[i]/(rho[i]*rho[j]))*(Pj1-Pi1);
			PIJW[2][j]=PIJW[2][j]+(mass[i]/(rho[i]*rho[j]))*(Pj2-Pi2);
		}
		if(itype[i]>0 && itype[j]<-10)
		{
			double PRii = p[i]+rho[i]*max(0,(0-PIJW[0][i])*vecn[0][j]+(0-PIJW[1][i])*vecn[1][j]+(0-PIJW[2][i])*vecn[2][j])
				*((x[0][i]-x[0][j])*vecn[0][j]+(x[1][i]-x[1][j])*vecn[1][j]+(x[2][i]-x[2][j])*vecn[2][j])
				*((x[0][i]-x[0][j])*vecn[0][j]+(x[1][i]-x[1][j])*vecn[1][j]+(x[2][i]-x[2][j])*vecn[2][j]);

			SForce[0][i] = SForce[0][i] - 2.0*(mass[j]/(rho[i]*rho[j]))*dwdx(x[0][i]-x[0][j],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[1][i] = SForce[1][i] - 2.0*(mass[j]/(rho[i]*rho[j]))*dwdx(x[1][i]-x[1][j],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[2][i] = SForce[2][i] - 2.0*(mass[j]/(rho[i]*rho[j]))*dwdx(x[2][i]-x[2][j],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[0][j] = SForce[0][j] - 2.0*(mass[i]/(rho[i]*rho[j]))*dwdx(x[0][j]-x[0][i],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[1][j] = SForce[1][j] - 2.0*(mass[i]/(rho[i]*rho[j]))*dwdx(x[1][j]-x[1][i],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
			SForce[2][j] = SForce[2][j] - 2.0*(mass[i]/(rho[i]*rho[j]))*dwdx(x[2][j]-x[2][i],dr)*((p[i]*rho[j]+PRii*rho[i])/(rho[i]+rho[j]));
		}
	}
	for(i=ntotal;i<ntotal+SolNum;i++)//速度
	{
		vx[0][i] = vx[0][i]+dt*PIJW[0][i]+dt*SForce[0][i];
		vx[1][i] = vx[1][i]+dt*PIJW[1][i]+dt*SForce[1][i];
		vx[2][i] = vx[2][i]+dt*PIJW[2][i]+dt*SForce[2][i]-dt*9.8;

		if(x[2][i]<=0.01 || x[1][i]<=0.01 || x[1][i]>=0.49)
		{
			vx[0][i]=0.0; vx[1][i]=0.0; vx[2][i]=0.0;
		}
	}
}
double max(double mi,double mj)
{
	if(mi>=mj)
	{return mi;}
	else
	{return mj;}
}
double min(double mi,double mj)
{
	if(mi>=mj)
	{return mj;}
	else
	{return mi;}
}

void NorVector()
{
	direct_find();
	for(i=0;i<num;i++)
	{
		sumvecn[0][i] = 0;
		sumvecn[1][i] = 0;
		sumvecn[2][i] = 0;
		SurF[i] = 0;
	}
	for(k=0;k<niac;k++)
	{
		i=pair_i.data()[k];
		j=pair_j.data()[k];
		double dr=sqrt((x[0][i]-x[0][j])*(x[0][i]-x[0][j])+(x[1][i]-x[1][j])*(x[1][i]-x[1][j])+(x[2][i]-x[2][j])*(x[2][i]-x[2][j])+0.000000000001);
		if(itype[i]<0 && itype[j]<0)
		{
			sumvecn[0][i] = sumvecn[0][i] - (mass[j]/rho[j])*dwdx(x[0][i]-x[0][j],dr);
			sumvecn[1][i] = sumvecn[1][i] - (mass[j]/rho[j])*dwdx(x[1][i]-x[1][j],dr);
			sumvecn[2][i] = sumvecn[2][i] - (mass[j]/rho[j])*dwdx(x[2][i]-x[2][j],dr);

			sumvecn[0][j] = sumvecn[0][j] - (mass[i]/rho[i])*dwdx(x[0][j]-x[0][i],dr);
			sumvecn[1][j] = sumvecn[1][j] - (mass[i]/rho[i])*dwdx(x[1][j]-x[1][i],dr);
			sumvecn[2][j] = sumvecn[2][j] - (mass[i]/rho[i])*dwdx(x[2][j]-x[2][i],dr);
		}
	}
	for(i=0;i<num;i++)
	{
		if(i<ntotal)
		{
			vecn[0][i] = 0;
			vecn[1][i] = 0;
			vecn[2][i] = 0;
		}
		else
		{
			vecn[0][i] = sumvecn[0][i]/sqrt(sumvecn[0][i]*sumvecn[0][i]+sumvecn[1][i]*sumvecn[1][i]+sumvecn[2][i]*sumvecn[2][i]+0.0000001);
			vecn[1][i] = sumvecn[1][i]/sqrt(sumvecn[0][i]*sumvecn[0][i]+sumvecn[1][i]*sumvecn[1][i]+sumvecn[2][i]*sumvecn[2][i]+0.0000001);
			vecn[2][i] = sumvecn[2][i]/sqrt(sumvecn[0][i]*sumvecn[0][i]+sumvecn[1][i]*sumvecn[1][i]+sumvecn[2][i]*sumvecn[2][i]+0.0000001);
			//vecn[0][i] = sumvecn[0][i]/fabs(sumvecn[0][i]);
			//vecn[1][i] = sumvecn[1][i]/fabs(sumvecn[1][i]);
			//vecn[2][i] = sumvecn[2][i]/fabs(sumvecn[2][i]);
		}
		dvxdt[0][i] = 0;
		dvxdt[1][i] = 0;
		dvxdt[2][i] = 0;
	}
}
