#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*    This file is part of the program and library ImitationMILP             *
#*    Copyright (C) 2018 Caltech                                             *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#@file    Makefile
#@brief   Makefile for Imitation MILP
#@author  Stephanie Ding

#-----------------------------------------------------------------------------
# include default project Makefile from SCIP
# Assumes SCIPDIR is defined as an environment variable
#-----------------------------------------------------------------------------

include $(SCIPDIR)/make/make.project

#-----------------------------------------------------------------------------
# Main Program
#-----------------------------------------------------------------------------
CC			=	gcc
CXX			= 	g++
MAINNAME	=	imilp
CMAINOBJ	=

CXXMAINOBJ	=	main.o \
				imitation_milp.o \
				eventhdlr_collectdata.o \
				eventhdlr_primalint.o \
				oracle.o \
				feat.o \
				ranked_pairs_collector.o \
				feat_computer_collector.o \
				nodesel_policy.o \
				oracle_scorer.o \
				python_scorer.o \
				ranknet_model.o


MAINSRC			=	$(addprefix $(SRCDIR)/,$(CMAINOBJ:.o=.c))
MAINSRC			+=	$(addprefix $(SRCDIR)/,$(CXXMAINOBJ:.o=.cpp))
MAINDEP			=	$(SRCDIR)/depend.cmain.$(OPT)

MAIN			=	$(MAINNAME).$(BASE).$(LPS)$(EXEEXTENSION)
MAINFILE		=	$(BINDIR)/$(MAIN)
MAINSHORTLINK	=	$(BINDIR)/$(MAINNAME)
MAINOBJFILES	=	$(addprefix $(OBJDIR)/,$(CMAINOBJ))
MAINOBJFILES	+=	$(addprefix $(OBJDIR)/,$(CXXMAINOBJ))

#-----------------------------------------------------------------------------
# External libraries
#-----------------------------------------------------------------------------
# location of the Python header files
PYTHON_VERSION = 3.8
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

FLAGS		+=	-g -shared -fPIC -O0 -std=c++11 -fno-lto -I$(BOOST_INC) -I$(PYTHON_INCLUDE)
FLAGS		+=	-Wfatal-errors $(filter-out -Wstrict-prototypes, $(shell python3-config --cflags))

LDFLAGS     +=  -ltorch -lc10 -lmkldnn -lgomp #-lnvToolsExt -lcaffe lcaffe_gpu
LDFLAGS		+=	-L$(BOOST_LIB) -lboost_system -lboost_filesystem -lboost_program_options -lboost_python3 -lpython$(PYTHON_VERSION) 
LDFLAGS		+=	$(shell python3-config --ldflags)

#-----------------------------------------------------------------------------
# Rules
#-----------------------------------------------------------------------------

ifeq ($(VERBOSE),false)
.SILENT:	$(MAINFILE) $(MAINOBJFILES) $(MAINSHORTLINK)
endif

.PHONY: all
all:		$(SCIPDIR) $(MAINFILE) $(MAINSHORTLINK)

.PHONY: lint
lint:	$(MAINSRC)
		-rm -f lint.out
		@echo $(FLAGS)
		$(SHELL) -ec 'for i in $^; \
			do \
			echo $$i; \
			$(LINT) $(SCIPDIR)/lint/scip.lnt +os\(lint.out\) -u -zero \
			$(FLAGS) -UNDEBUG -UWITH_READLINE -UROUNDING_FE $$i; \
			done'
.PHONY: scip
scip:
		@$(MAKE) -C $(SCIPDIR) libs $^

.PHONY: doc
doc:
		@-(cd doc && ln -fs ../$(SCIPDIR)/doc/scip.css);
		@-(cd doc && ln -fs ../$(SCIPDIR)/doc/pictures/scippy.png);
		@-(cd doc && ln -fs ../$(SCIPDIR)/doc/pictures/miniscippy.png);
		@-(cd doc && ln -fs ../$(SCIPDIR)/doc/scipfooter.html footer.html);
		cd doc; $(DOXY) $(MAINNAME).dxy

$(MAINSHORTLINK):	$(MAINFILE)
		@rm -f $@
		cd $(dir $@) && ln -s $(notdir $(MAINFILE)) $(notdir $@)

$(OBJDIR):
		@-mkdir -p $(OBJDIR)

$(BINDIR):
		@-mkdir -p $(BINDIR)

.PHONY: clean
clean:		$(OBJDIR)
ifneq ($(OBJDIR),)
		@-(rm -rf $(OBJDIR)/*.o && rmdir $(OBJDIR));
		@echo "-> remove main object files"
endif
		@-rm -f $(MAINFILE) $(MAINLINK) $(MAINSHORTLINK)
		@echo "-> remove binary"

.PHONY: test
test:           $(MAINFILE)
		@-(cd check && ln -fs ../$(SCIPDIR)/check/check.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/evalcheck.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/evalcheck_cluster.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/check.awk);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/getlastprob.awk);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/configuration_set.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/configuration_logfiles.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/configuration_tmpfile_setup_scip.sh);
		@-(cd check && ln -fs ../$(SCIPDIR)/check/run.sh);
		cd check; \
		$(SHELL) ./check.sh $(TEST) $(MAINFILE) $(SETTINGS) $(notdir $(MAINFILE)) $(TIME) $(NODES) $(MEM) $(THREADS) $(FEASTOL) $(DISPFREQ) $(CONTINUE) $(LOCK) "example" $(LPS) $(VALGRIND) $(CLIENTTMPDIR) $(OPTCOMMAND);

.PHONY: tags
tags:
		rm -f TAGS; ctags -e src/*.c src/*.h $(SCIPDIR)/src/scip/*.c $(SCIPDIR)/src/scip/*.h;

.PHONY: depend
depend:		$(SCIPDIR)
		$(SHELL) -ec '$(DCC) $(FLAGS) $(DFLAGS) $(MAINSRC) \
		| sed '\''s|^\([0-9A-Za-z\_]\{1,\}\)\.o *: *$(SRCDIR)/\([0-9A-Za-z\_]*\).c|$$\(OBJDIR\)/\2.o: $(SRCDIR)/\2.c|g'\'' \
		>$(MAINDEP)'

-include	$(MAINDEP)

$(MAINFILE):	$(BINDIR) $(OBJDIR) $(SCIPLIBFILE) $(LPILIBFILE) $(NLPILIBFILE) $(MAINOBJFILES)
		@echo "-> linking $@"
		$(LINKCXX) $(MAINOBJFILES) $(LINKCXXSCIPALL) \
		$(LINKCXX_L)$(SCIPDIR)/lib $(LINKCXX_l)$(SCIPLIB)$(LINKLIBSUFFIX) \
                $(LINKCXX_l)$(LPILIB)$(LINKLIBSUFFIX) $(LINKCXX_l)$(NLPILIB)$(LINKLIBSUFFIX) \
                $(OFLAGS) $(LPSLDFLAGS) \
		$(LDFLAGS) $(LINKCXX_o)$@

$(OBJDIR)/%.o:	$(SRCDIR)/%.c
		@echo "-> compiling $@"
		$(CC) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CFLAGS) -c $< $(CC_o)$@ -I$(SCIPDIR)/src/

$(OBJDIR)/%.o:	$(SRCDIR)/%.cpp
		@echo "-> compiling $@"
		$(CXX) $(FLAGS) $(OFLAGS) $(BINOFLAGS) $(CXXFLAGS) -c $< $(CXX_o)$@

#---- EOF --------------------------------------------------------------------
